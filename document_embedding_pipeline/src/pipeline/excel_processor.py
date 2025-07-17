"""
Excel processing pipeline that plugs into the existing state-machine.
Keeps the rich parsing logic of the GitHub ExcelParser, including
complex table finding.
"""
from __future__ import annotations
import asyncio
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
    ):
        self.llm = llm_service or LLMService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def process(self, file_path: Path) -> Dict[str, Any]:
        """
        Full async pipeline for a single Excel file.
        Returns the same dict structure that WordProcessor returns so the
        orchestrator can treat both uniformly, but enriched with metadata
        and detailed cell info from ExcelParser.
        """
        logger.info("Starting Excel processing: %s", file_path.name)

        workbook = load_workbook(file_path, data_only=True)
        try:
            sheets_out = []
            total_rows = 0
            total_cells_with_content = 0

            for sheet_name in workbook.sheetnames:
                logger.info(f"Processing sheet: {sheet_name} in {file_path.name}")
                sheet = workbook[sheet_name]
                sheet_result = await self._process_sheet(sheet)
                sheets_out.append(sheet_result)
                
                total_rows += sheet.max_row
                for table in sheet_result["tables"]:
                    for row in table["rows"]:
                        total_cells_with_content += sum(1 for v in row["raw_data"].values() if v and v.get("value") is not None)

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
    async def _process_sheet(self, sheet) -> Dict[str, Any]:
        """
        Parse one sheet, detect tables, generate summaries + row texts.
        """
        # This now uses the more complex table finding logic
        tables = self._find_tables(sheet)

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
                    # Ensure we don't go out of bounds if texts is shorter
                    if row_cells:
                        rows_out.append(
                            {
                                "raw_data": {h: cell for h, cell in zip(headers, row_cells)},
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

        return {"sheet_name": sheet.title, "tables": processed_tables}

    # REPLACED: This method now uses the more complex logic from ExcelParser.
    def _find_tables(self, sheet) -> List[Dict[str, Any]]:
        """Find and process all tables in a sheet using a more complex algorithm."""
        tables = []
        visited = set()  # Track already processed cells

        def get_table(start_row, start_col):
            """Extract a table starting from (start_row, start_col)."""
            # Find the last column of the table by finding the rightmost column with data
            max_col = start_col
            for col in range(start_col, sheet.max_column + 1):
                has_data_in_col = False
                for r in range(start_row, sheet.max_row + 1):
                    if sheet.cell(row=r, column=col).value is not None:
                        has_data_in_col = True
                        max_col = col
                        break
                if not has_data_in_col and col > start_col:
                     # Stop if we find a fully empty column past the first one
                    break
            
            # Find the last row of the table
            max_row = start_row
            for row in range(start_row, sheet.max_row + 1):
                has_data_in_row = any(sheet.cell(row=row, column=c).value is not None for c in range(start_col, max_col + 1))
                if has_data_in_row:
                    max_row = row
                else:
                    break

            # Process the rectangular table region
            headers = []
            data_rows = []

            # Process header row
            header_cells = []
            for c in range(start_col, max_col + 1):
                cell = sheet.cell(row=start_row, column=c)
                header_value = self._enrich_cell(cell, None, start_row, c)
                header_cells.append(header_value)
                if cell.value is not None:
                    visited.add((start_row, c))
            
            if not any(cell["value"] is not None for cell in header_cells):
                return None # Not a valid table if header is empty

            headers = [str(cell["value"] or "") for cell in header_cells]

            # Process data rows
            for r in range(start_row + 1, max_row + 1):
                row_data = []
                for c in range(start_col, max_col + 1):
                    cell = sheet.cell(row=r, column=c)
                    header = headers[c - start_col] if c - start_col < len(headers) else None
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
                }
            }

        # Find all tables in the sheet
        for r_idx in range(1, sheet.max_row + 1):
            for c_idx in range(1, sheet.max_column + 1):
                if (r_idx, c_idx) in visited:
                    continue

                cell = sheet.cell(row=r_idx, column=c_idx)
                # A potential table starts with a non-visited, non-empty string cell
                if cell.value and isinstance(cell.value, str):
                    table = get_table(r_idx, c_idx)
                    if table and table["data"]:
                        tables.append(table)
        
        return tables

    def _enrich_cell(self, cell, header: str, row: int, col: int) -> Dict[str, Any]:
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
                    "bold": cell.font.bold,
                    "italic": cell.font.italic,
                    "size": cell.font.size,
                    "color": cell.font.color.rgb if cell.font.color else None,
                },
                "fill": {
                    "background_color": (
                        cell.fill.start_color.rgb if cell.fill.start_color else None
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

        return enriched_data

    # ---------------- LLM calls ---------------------------------------

    async def _rewrite_headers(self, table: Dict[str, Any]) -> List[str]:
        """Use LLM to clean / rewrite column headers."""
        sample_rows = [[c["value"] for c in r] for r in table["data"][:3]]
        prompt = (
            "You are a data-analysis expert.\n"
            f"Original headers: {table['headers']}\n"
            f"First 3 rows: {sample_rows}\n\n"
            "Return only the new comma-separated headers, same order, cleaned."
        )
        resp : Headers = await self.llm.aget_structured_response(prompt, max_tokens=256, response_model=Headers)
        new_headers = resp.headers
        return new_headers if len(new_headers) == len(table["headers"]) else table["headers"]

    async def _summarise_table(self, headers: List[str], data: List[List[Dict[str, Any]]]) -> str:
        """Generate a concise natural-language summary of the table."""
        sample = [{h: str(cell["value"]) for h, cell in zip(headers, row)} for row in data[:3]]
        prompt = (
            "Summarise the following table in 1-2 sentences:\n"
            f"Headers: {headers}\n"
            f"Sample rows: {json.dumps(sample, ensure_ascii=False, indent=2)}"
        )
        resp : Summary = await self.llm.aget_structured_response(prompt, max_tokens=256, response_model=Summary)
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
            resp : SerializedRows = await self.llm.aget_structured_response(prompt, max_tokens=2048, response_model=SerializedRows)
            return resp.rows
        except Exception as e:
            logger.error(f"Could not generate structured text for rows. Error: {e}")
            return [""] * len(rows)