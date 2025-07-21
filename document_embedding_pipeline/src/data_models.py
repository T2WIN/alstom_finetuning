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
    table_schema : List[Column]
    table_summary : str

Sheet = Annotated[
    Union[ContentSheet, TableSheet],
    Field(discriminator="sheet_type")
]

class Header(BaseModel):
    name : str = Field(..., description="The name of header exactly as seen in the table")
    description : str = Field(..., description="A text description of the header infered on the content of rows")

class Headers(BaseModel):
    headers : List[Header] = Field(..., description="A list of clean headers")

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

class HierarchicalNode(BaseModel):
    """A recursive model for deeply nested documents, representing a node in a tree."""
    node_type: Literal["Section", "Table"] = Field(description="The type of the node, either a text section or a table.")
    title: str = Field(description="The title of this section or the caption of the table.")
    children: List['HierarchicalNode'] = Field(default_factory=list, description="A list of child nodes.")

# Update the forward reference
HierarchicalNode.model_rebuild()

class WordDocumentStructure(BaseModel):

    title: str = Field(description="The title of the entire document, often found in the first page, otherwise infer it from the document content")
    structure : List['HierarchicalNode'] = Field(description="The hierarchical structure of the document")