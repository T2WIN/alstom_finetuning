
from pydantic import BaseModel, Field
from enum import Enum
from typing import List

class Query(BaseModel):
    query: str = Field(
        ...,
        description="A query for retrieving the passage"
    )

class QueryType(str, Enum):
    FACT_RETRIEVAL = "Fact Retrieval"
    ANALYTICAL = "Analytical/Explanatory"
    REASONING = "Reasoning"
    HYPOTHETICAL = "Hypothetical/Scenario-Based"
    SUMMARIZATION = "Summarization"
    EVALUATIVE = "Evaluative"

class QueryFormat(str, Enum):
    KEYWORDS = "keywords"
    FORMAL_QUESTION = "formal question"

class CharacterQuestionAnalysis(BaseModel):
    query_type: QueryType
    query_format: QueryFormat

class ConfigurationSet(BaseModel):
    configurations: List[CharacterQuestionAnalysis] = Field(
        ...,
        max_length=5,
        description="A list of up to 5 query configurations."
    )

class QuerySet(BaseModel):
    queries: List[Query] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="A list containing exactly 5 queries."
    )