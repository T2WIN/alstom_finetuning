from typing import List, Optional, Union, Literal, Annotated
import abc
from pydantic import BaseModel, Field
from pathlib import Path

class Section(BaseModel):
    """
    Represents a section within a Word document.
    """
    title:str
    content: str
    summary: str
    table_summary: Optional[str] = None

class WordDocumentPayload(BaseModel):
    """
    Represents the payload for a Word document.
    """
    file_path: Path
    title: str
    global_summary: str
    sections: List[Section]


class BaseSheet(BaseModel, abc.ABC):
    """
    Represents a sheet within an Excel document.
    """
    sheet_name: str

class ContentSheet(BaseSheet):
    """
    Represents a sheet within an Excel document.
    """
    sheet_type: Literal["content"] = "content"
    content : str
    summary : str

class Column(BaseModel):
    column_name: str
    description: str

class TableSheet(BaseSheet):
    sheet_type: Literal["table"] = "table"
    table_schema : List[Column]
    table_summary : str

Sheet = Annotated[
    Union[ContentSheet, TableSheet],
    Field(discriminator="sheet_type")
]


class ExcelDocumentPayload(BaseModel):
    """
    Represents the payload for an Excel document.
    """
    title:str
    file_path: Path
    sheets: List[Sheet]
