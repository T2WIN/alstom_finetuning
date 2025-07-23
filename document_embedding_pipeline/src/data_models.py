from typing import List, Optional, Union, Literal, Annotated, Dict
import abc
from pydantic import BaseModel, Field
from pathlib import Path


class BaseSheet(BaseModel, abc.ABC):
    """
    Represents a sheet within an Excel document.
    """
    sheet_name: str

class ContentSheet(BaseSheet):
    """
    Represents a sheet within an Excel document.
    """
    sheet_name : str
    sheet_type: Literal["content"] = "content"
    content : str
    summary : str

class Column(BaseModel):
    name: str
    description: str

class TableSheet(BaseSheet):
    sheet_name : str
    sheet_type: Literal["table"] = "table"
    table_headers : List[str]
    table_summary : str
    serialized_rows: List[Dict]

Sheet = Annotated[
    Union[ContentSheet, TableSheet],
    Field(discriminator="sheet_type")
]

class Header(BaseModel):
    name : str = Field(..., description="The name of header exactly as seen in the table")
    description : str = Field(..., description="A text description of the header infered on the content of rows")

class Headers(BaseModel):
    headers : List[Header] = Field(..., description="A list of clean individual headers")

class Summary(BaseModel):
    summary : str = Field(..., description="A summary of the table")

class SerializedRows(BaseModel):
    rows : List[str] = Field(..., description="List of serialized rows as sentences")

class ExcelDocumentPayload(BaseModel):
    """
    Represents the payload for an Excel document.
    """
    title:str
    file_path: str
    sheets: List[Sheet]

class Title(BaseModel):
    title: Optional[str] = None


class Section(BaseModel):
    """
    A model to represent a section of a document, which can contain nested subsections.
    """
    title: str = Field(..., description="The verbatim name of the section")
    content_summary: str = Field(..., description="A 1-2 sentence summary of this section. This summary should present the purpose of the section not just what is there.")
    subsections: Optional[List['Section']] = Field(default=None, description="The subsections of that section")

Section.model_rebuild()

class WordDocumentStructure(BaseModel):
    structure : List['Section'] = Field(description="The hierarchical structure of the document")