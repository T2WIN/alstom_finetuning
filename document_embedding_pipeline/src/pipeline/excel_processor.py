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
from data_models import Headers, Summary, SerializedRows, ExcelDocumentPayload, Header


logger = logging.getLogger(__name__)

class ExcelProcessor(BaseProcessor):
    """
    Excel equivalent of WordProcessor.
    Implements:
        process(file_path: Path) -> Dict[str, Any]
    """

    def __init__(
        self,
        temp_folder : Path,
    ):
        """
        Initializes the ExcelProcessor.

        Args:
            temp_folder : Path
        """
        self.llm = LLMService()

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
            content_sheets_data = []
            table_sheets_data = []
            total_rows = 0
            total_cells_with_content = 0
            classifications = []
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                classification = self._classify_sheet(sheet)
                logger.info(f"Sheet: {sheet_name} is {classification}")
                classifications.append((sheet_name, classification))

            for sheet_name, classification in classifications:
                if classification == "content":
                    sheet = workbook[sheet_name]
                    logger.info(f"Parsing content sheet {sheet_name}")
                    content = "" 
                    blocks = self._find_all_blocks(sheet)
                    for idx, block in enumerate(blocks):
                        content += f"Block {idx} :"
                        content += self._tiny_markdown_table(sheet, block)
                        content += "\n"
                    summary = "summary" #Placeholder for now, will be turned into LLM call later
                    content_sheets_data.append((sheet_name, content, summary))

            for sheet_name, classification in classifications:       
                if classification == "table":
                    sheet = workbook[sheet_name]
                    logger.info(f"Parsing table sheet {sheet_name}")
                    # 1. Find tables using the existing logic
                    raw_tables = self._find_tables(sheet)

                    # 3. Process the found tables to generate summaries and text
                    logger.info(content_sheets_data)
                    sheet_result = await self._process_found_tables(
                        sheet.title, raw_tables, content_sheets_data
                    )
                    table_sheets_data.append(sheet_result)

                    # Update stats
                    total_rows += sheet.max_row
                    for table in sheet_result["tables"]:
                        for row in table["rows"]:
                            total_cells_with_content += sum(
                                1
                                for v in row["raw_data"].values()
                                if v and v.get("value") is not None
                            )                  

            content_sheets_cleaned = []
            for sheet_name, full_text_content, summary in content_sheets_data:
                content_sheets_cleaned.append({
                    "sheet_type" : "content",
                    "sheet_name": sheet_name,
                    "summary" : summary,
                    "content" : full_text_content
                })
            
            table_sheets_cleaned = []
            # logger.info(table_sheets_data[0]["tables"])
            for idx, sheet in enumerate(table_sheets_data):
                for table in sheet["tables"]:
                    headers = [header.model_dump() for header in table["headers"]]
                    table_sheets_cleaned.append({
                        "sheet_type" : "table",
                        "sheet_name" : f"{sheet_name} {idx}",
                        "table_schema" : headers,
                        "table_summary": table["summary"],
                    })
            logger.info(table_sheets_cleaned[0])
            data = {
                "file_path" : str(file_path),
                "title" : "test",
                "sheets" : table_sheets_cleaned 
            }
            return ExcelDocumentPayload(**data)

        finally:
            workbook.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------   -------------------------------

    @staticmethod
    def _tiny_markdown_table(ws, block, rows=8, cols=6):
        """
        Returns a compact markdown table (no header, ≤ rows×cols cells)
        for the top-left corner of the block.
        """
        r1, r2 = block["r1"], block["r2"]
        c1, c2 = block["c1"], block["c2"]
        lines = []
        for r in range(r1, min(r1 + rows, r2 + 1)):
            cells = [
                str(ws.cell(row=r, column=c).value or "")
                for c in range(c1, min(c1 + cols, c2 + 1))
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if r2 - r1 + 1 > rows:
            lines[-1] += "\n⋮"
        if c2 - c1 + 1 > cols:
            lines = [ln + " …" for ln in lines]
        return "\n".join(lines) if lines else "(empty)"

    def _classify_sheet(self, sheet) -> str:
        """
        Classifies the worksheet as either 'table' or 'content'.

        Returns
        -------
        str
            'table'  – if the sheet contains at least one block with > 100 filled cells  
            'content' – otherwise
        """
        blocks: List[Dict] = self._find_all_blocks(sheet)

        # Check the “table” criterion
        if any(b["non_empty"] > 80 for b in blocks):
            return "table"

        return "content"

    @staticmethod
    def _find_all_blocks(ws):
        """
        Quick, lightweight scan to return every rectangular block of
        non-empty cells (including single-cell blocks).
        Returns list[dict] with r1, r2, c1, c2, rows, cols, non_empty.
        """
        visited = [[False] * (ws.max_column + 1) for _ in range(ws.max_row + 1)]
        blocks = []

        def _dfs(r, c):
            stack = [(r, c)]
            min_r = max_r = r
            min_c = max_c = c
            filled = 0
            while stack:
                x, y = stack.pop()
                if x < 1 or y < 1 or x > ws.max_row or y > ws.max_column:
                    continue
                if visited[x][y] or ws.cell(row=x, column=y).value is None:
                    continue
                visited[x][y] = True
                filled += 1
                min_r, max_r = min(min_r, x), max(max_r, x)
                min_c, max_c = min(min_c, y), max(max_c, y)
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    stack.append((x + dx, y + dy))
            return {
                "r1": min_r, "r2": max_r, "c1": min_c, "c2": max_c,
                "rows": max_r - min_r + 1,
                "cols": max_c - min_c + 1,
                "non_empty": filled,
            }

        for row in range(1, min(ws.max_row + 1, 200)): # min is there to avoid visiting all cells in very large tables
            for col in range(1, min(ws.max_column + 1, 100)):
                if not visited[row][col] and ws.cell(row=row, column=col).value is not None:
                    blocks.append(_dfs(row, col))
        return blocks
    
    async def _process_found_tables(
        self, sheet_name: str, tables: List[Dict[str, Any]], context : List[(str,str, str)]
    ) -> Dict[str, Any]:
        """
        Parse a list of found tables, generate summaries + row texts.
        """
        processed_tables: List[Dict[str, Any]] = []
        for tbl in tables:
            headers : Headers = await self._rewrite_headers(tbl)
            summary = await self._summarise_table(headers, tbl["data"], context)

            rows_out: List[Dict[str, Any]] = []
            # batch_size = 20
            # for start in range(0, len(tbl["data"]), batch_size):
            #     batch = tbl["data"][start : start + batch_size]
            #     texts = await self._rows_to_text(headers, summary, batch)
            #     for i, (row_cells, nl_text) in enumerate(zip(batch, texts)):
            #         if row_cells:
            #             rows_out.append(
            #                 {
            #                     "raw_data": {
            #                         h: cell for h, cell in zip(headers, row_cells)
            #                     },
            #                     "natural_language_text": nl_text,
            #                     "row_num": row_cells[0]["row"],
            #                 }
            #             )

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
        sample_rows = [[c["value"] for c in r] for r in table["data"][4:7]]
        prompt = (
            "You are a data-analysis expert.\n"
            f"Original headers: {table['headers']}\n"
            f"First 3 rows: {sample_rows}\n\n"
            "Return only the new headers, same order, cleaned with a short description based on the values seen in the rows."
        )
        headers: Headers = await self.llm.aget_structured_response(
            prompt, max_tokens=1024, response_model=Headers
        )
        return headers.headers

    async def _summarise_table(
        self, headers: List[Header], data: List[List[Dict[str, Any]]], context : List[(str, str, str)]
    ) -> str:
        """Generate a concise natural-language summary of the table."""
        formatted_context = ""
        for idx, (sheet_name, sheet_content, _) in enumerate(context):
            formatted_context += f"Sheet {idx} called {sheet_name} :\n"
            formatted_context += sheet_content
            formatted_context += "\n"

        sample = [
            {h.name: str(cell["value"]) for h, cell in zip(headers, row)}
            for row in data[4:7]
        ]

        header_names = [header.name for header in headers]
        prompt = (
            "Summarise the following table in 1-2 sentences:\n"
            f"Headers: {header_names}\n"
            f"Sample rows: {json.dumps(sample, ensure_ascii=False, indent=2)}"
            f"Here is context gathered from other sheets that might help you understand : {formatted_context}"
        )
        resp: Summary = await self.llm.aget_structured_response(
            prompt, max_tokens=512, response_model=Summary
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
            "Output in JSON"
        )
        try:
            resp: SerializedRows = await self.llm.aget_structured_response(
                prompt, response_model=SerializedRows
            )
            return resp.rows
        except Exception as e:
            logger.error(f"Could not generate structured text for rows. Error: {e}")
            return [""] * len(rows)