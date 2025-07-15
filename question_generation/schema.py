
from pydantic import BaseModel, Field
from enum import Enum

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