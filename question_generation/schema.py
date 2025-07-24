
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from typing import List

class QuerySet(BaseModel):
    queries: list[str] = Field(
        ...,
        description="A list of queries for retrieving the passage"
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
    query_type: QueryType = Field(..., description="The type of query to ask")
    query_format: QueryFormat = Field(..., description="The format of the query, keyword or formal")

class ConfigSet(BaseModel):
    configs : List[CharacterQuestionAnalysis]