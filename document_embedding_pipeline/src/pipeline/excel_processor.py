"""
Excel processing pipeline that plugs into the existing state-machine.
Keeps the rich parsing logic of the GitHub ExcelParser, including
complex table finding.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from openpyxl import load_workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.utils import get_column_letter

from services.llm_service import LLMService
from utils.logging_setup import setup_logging
from pipeline.base_processor import BaseProcessor
from data_models import Headers, Summary, SerializedRows


logger = logging.getLogger(__name__)

class ExcelProcessor(BaseProcessor):
    """
    Excel equivalent of WordProcessor.
    Implements:
        process(file_path: Path) -> Dict[str, Any]
    """

    def __init__(
        self,
        llm_service: LLMService | None = None,
        debug_output_path: Path | None = None,  # MODIFIED: Added debug path
    ):
        """
        Initializes the ExcelProcessor.

        Args:
            llm_service: An instance of LLMService.
            debug_output_path: If provided, saves the raw detected tables
                               to a JSON file in this directory for evaluation.
        """
        self.llm = llm_service or LLMService()
        self.debug_output_path = debug_output_path  # MODIFIED: Store debug path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Full async pipeline for a single Excel file.
        Returns the same dict structure that WordProcessor returns so the
        orchestrator can treat both uniformly, but enriched with metadata
        and detailed cell info from ExcelParser.

        MODIFIED: Now also saves raw table detection results to a JSON
        file if `debug_output_path` is set.
        """
        logger.info("Starting Excel processing: %s", file_path.name)

        workbook = load_workbook(file_path, data_only=True)
        # NEW: List to store raw table data for debugging
        debug_sheets_data = []
        try:
            sheets_out = []
            total_rows = 0
            total_cells_with_content = 0

            for sheet_name in workbook.sheetnames:
                logger.info(f"Processing sheet: {sheet_name} in {file_path.name}")
                sheet = workbook[sheet_name]

                # 1. Find tables using the existing logic
                raw_tables = self._find_tables(sheet)

                # 2. If debugging is enabled, store the raw results
                if self.debug_output_path:
                    debug_sheets_data.append(
                        {"sheet_name": sheet_name, "detected_tables": raw_tables}
                    )

                # 3. Process the found tables to generate summaries and text
                sheet_result = await self._process_found_tables(
                    sheet.title, raw_tables
                )
                sheets_out.append(sheet_result)

                # Update stats
                total_rows += sheet.max_row
                for table in sheet_result["tables"]:
                    for row in table["rows"]:
                        total_cells_with_content += sum(
                            1
                            for v in row["raw_data"].values()
                            if v and v.get("value") is not None
                        )

            all_text: List[str] = []
            for sh in sheets_out:
                for tbl in sh["tables"]:
                    all_text.extend(r["natural_language_text"] for r in tbl["rows"])

            props = workbook.properties
            metadata = {
                "creator": props.creator,
                "created": props.created.isoformat() if props.created else None,
                "modified": props.modified.isoformat() if props.modified else None,
                "last_modified_by": props.lastModifiedBy,
                "sheet_count": len(workbook.sheetnames),
            }

            # 4. NEW: Save the debug file after processing all sheets
            if self.debug_output_path:
                self._save_debug_json(file_path, debug_sheets_data)

            return {
                "file": str(file_path),
                "metadata": metadata,
                "sheets": sheets_out,
                "text_content": "\n".join(all_text),
                "sheet_names": workbook.sheetnames,
                "total_rows": total_rows,
                "total_cells": total_cells_with_content,
            }

        finally:
            workbook.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # NEW: Helper function to save debug data
    def _save_debug_json(
        self, original_path: Path, sheets_data: List[Dict[str, Any]]
    ):
        """Saves the raw detected tables to a JSON file for debugging."""
        if not self.debug_output_path:
            return

        self.debug_output_path.mkdir(parents=True, exist_ok=True)
        output_filename = original_path.stem + "_tables.json"
        output_file_path = self.debug_output_path / output_filename

        debug_data = {"source_file": str(original_path), "sheets": sheets_data}

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved table detection debug info to: %s", output_file_path)
        except TypeError as e:
            logger.error(
                f"Could not serialize debug data to JSON. Check for non-serializable types. Error: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to save debug JSON: {e}")

    # REFACTORED: The original _process_sheet logic is now in this new helper
    # that accepts a list of tables directly.
    async def _process_found_tables(
        self, sheet_name: str, tables: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse a list of found tables, generate summaries + row texts.
        """
        processed_tables: List[Dict[str, Any]] = []
        for tbl in tables:
            headers = await self._rewrite_headers(tbl)
            summary = await self._summarise_table(headers, tbl["data"])

            rows_out: List[Dict[str, Any]] = []
            batch_size = 20
            for start in range(0, len(tbl["data"]), batch_size):
                batch = tbl["data"][start : start + batch_size]
                texts = await self._rows_to_text(headers, summary, batch)
                for i, (row_cells, nl_text) in enumerate(zip(batch, texts)):
                    if row_cells:
                        rows_out.append(
                            {
                                "raw_data": {
                                    h: cell for h, cell in zip(headers, row_cells)
                                },
                                "natural_language_text": nl_text,
                                "row_num": row_cells[0]["row"],
                            }
                        )

            processed_tables.append(
                {
                    "headers": headers,
                    "summary": summary,
                    "rows": rows_out,
                    "location": tbl["location"],
                }
            )

        return {"sheet_name": sheet_name, "tables": processed_tables}

    # NOTE: The original _process_sheet function was removed and its logic
    # was refactored into the `process` and `_process_found_tables` methods.

    # This method remains unchanged.
    def _find_tables(self, sheet) -> List[Dict[str, Any]]:
        """Find and process all tables in a sheet using a more complex algorithm."""
        tables = []
        visited = set()  # Track already processed cells

        def get_table(start_row, start_col):
            """Extract a table starting from (start_row, start_col)."""
            max_col = start_col
            for col in range(start_col, sheet.max_column + 1):
                has_data_in_col = False
                for r in range(start_row, sheet.max_row + 1):
                    if sheet.cell(row=r, column=col).value is not None:
                        has_data_in_col = True
                        max_col = col
                        break
                if not has_data_in_col and col > start_col:
                    break

            max_row = start_row
            for row in range(start_row, sheet.max_row + 1):
                has_data_in_row = any(
                    sheet.cell(row=row, column=c).value is not None
                    for c in range(start_col, max_col + 1)
                )
                if has_data_in_row:
                    max_row = row
                else:
                    break

            header_cells = []
            for c in range(start_col, max_col + 1):
                cell = sheet.cell(row=start_row, column=c)
                header_value = self._enrich_cell(cell, None, start_row, c)
                header_cells.append(header_value)
                if cell.value is not None:
                    visited.add((start_row, c))

            if not any(cell["value"] is not None for cell in header_cells):
                return None

            headers = [str(cell["value"] or "") for cell in header_cells]

            data_rows = []
            for r in range(start_row + 1, max_row + 1):
                row_data = []
                for c in range(start_col, max_col + 1):
                    cell = sheet.cell(row=r, column=c)
                    header = (
                        headers[c - start_col]
                        if c - start_col < len(headers)
                        else None
                    )
                    cell_data = self._enrich_cell(cell, header, r, c)
                    if cell.value is not None:
                        visited.add((r, c))
                    row_data.append(cell_data)
                data_rows.append(row_data)

            return {
                "headers": headers,
                "data": data_rows,
                "location": {
                    "start_row": start_row,
                    "start_col": start_col,
                    "end_row": max_row,
                    "end_col": max_col,
                },
            }

        for r_idx in range(1, sheet.max_row + 1):
            for c_idx in range(1, sheet.max_column + 1):
                if (r_idx, c_idx) in visited:
                    continue

                cell = sheet.cell(row=r_idx, column=c_idx)
                if cell.value and isinstance(cell.value, str):
                    table = get_table(r_idx, c_idx)
                    if table and table["data"]:
                        tables.append(table)

        return tables

    def stringify_dict_values(self, d):
        """
        Recursively iterates through a dictionary and applies the str() function
        to all non-dictionary values.

        This function handles nested dictionaries, ensuring that every value
        at every level of nesting is converted to a string.

        Args:
            d (dict): The dictionary to process.

        Returns:
            dict: A new dictionary with all values converted to strings.
        """
        if not isinstance(d, dict):
            return d

        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                new_dict[key] = self.stringify_dict_values(value)
            else:
                # Otherwise, convert the value to a string
                new_dict[key] = str(value)
        return new_dict

    # This method remains unchanged.
    def _enrich_cell(
        self, cell, header: str, row: int, col: int
    ) -> Dict[str, Any]:
        """Turn openpyxl cell into a serialisable dict with rich metadata."""
        value = cell.value
        data_type = cell.data_type

        if isinstance(cell, MergedCell):
            data_type = "merged"
            for rng in cell.parent.merged_cells.ranges:
                if cell.coordinate in rng:
                    value = cell.parent.cell(rng.min_row, rng.min_col).value
                    break

        enriched_data = {
            "value": value,
            "header": header,
            "row": row,
            "column": col,
            "column_letter": get_column_letter(col),
            "coordinate": f"{get_column_letter(col)}{row}",
            "data_type": data_type,
            "style": {
                "font": {
                    "bold": str(cell.font.bold),
                    "italic": cell.font.italic,
                    "size": cell.font.size,
                    "color": str(cell.font.color.rgb) if cell.font.color else None,
                },
                "fill": {
                    "background_color": (
                        str(cell.fill.start_color.rgb) if cell.fill.start_color else None
                    )
                },
                "alignment": {
                    "horizontal": cell.alignment.horizontal,
                    "vertical": cell.alignment.vertical,
                },
            },
        }

        if cell.data_type == "f":
            enriched_data["formula"] = cell.value

        return self.stringify_dict_values(enriched_data)

    # ---------------- LLM calls (unchanged) -------------------------

    async def _rewrite_headers(self, table: Dict[str, Any]) -> List[str]:
        """Use LLM to clean / rewrite column headers."""
        sample_rows = [[c["value"] for c in r] for r in table["data"][:3]]
        prompt = (
            "You are a data-analysis expert.\n"
            f"Original headers: {table['headers']}\n"
            f"First 3 rows: {sample_rows}\n\n"
            "Return only the new comma-separated headers, same order, cleaned."
        )
        resp: Headers = await self.llm.aget_structured_response(
            prompt, max_tokens=256, response_model=Headers
        )
        new_headers = resp.headers
        return (
            new_headers
            if len(new_headers) == len(table["headers"])
            else table["headers"]
        )

    async def _summarise_table(
        self, headers: List[str], data: List[List[Dict[str, Any]]]
    ) -> str:
        """Generate a concise natural-language summary of the table."""
        sample = [
            {h: str(cell["value"]) for h, cell in zip(headers, row)}
            for row in data[:3]
        ]
        prompt = (
            "Summarise the following table in 1-2 sentences:\n"
            f"Headers: {headers}\n"
            f"Sample rows: {json.dumps(sample, ensure_ascii=False, indent=2)}"
        )
        resp: Summary = await self.llm.aget_structured_response(
            prompt, max_tokens=256, response_model=Summary
        )
        return resp.summary

    async def _rows_to_text(
        self,
        headers: List[str],
        table_summary: str,
        rows: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """Convert a batch of rows into natural language sentences."""
        if not rows:
            return []

        rows_dict = [{h: str(c["value"]) for h, c in zip(headers, r)} for r in rows]
        prompt = (
            "Given the table summary below, convert each row into a standalone sentence.\n"
            f"Summary: {table_summary}\n"
            f"Rows: {json.dumps(rows_dict, ensure_ascii=False, indent=2)}\n\n"
            "Return a list of strings, where each string corresponds to a row."
        )
        try:
            resp: SerializedRows = await self.llm.aget_structured_response(
                prompt, response_model=SerializedRows
            )
            return resp.rows
        except Exception as e:
            logger.error(f"Could not generate structured text for rows. Error: {e}")
            return [""] * len(rows)