from pydantic import BaseModel, Field
from typing import List

class QueryEthics(BaseModel):
    is_ethical: bool = Field(default=True, description="Is the query ethical?")
    confidence: float = Field(default=1.0, description="Confidence level of the ethics evaluation")
    category: str = Field(default="", description="Category of ethical concern if any")
    reason: str = Field(default="", description="Reason if the query is not ethical")

class QueryMedicine(BaseModel):
    is_medicine_related: bool = Field(default=False, description="Is the query related to medicine?")    

class RagState(BaseModel):
    found_info: bool = Field(default=False, description="Whether relevant information was found")
    context: str = Field(default="", description="Contextual information retrieved")
    response: str = Field(default="", description="Response generated based on the context")

class AppState(BaseModel):
    query: str = Field(default="", description="User query")
    ethics: QueryEthics = Field(default=QueryEthics(), description="Ethics check result")
    medicine: QueryMedicine = Field(default=QueryMedicine(), description="Medicine relevance check result")
    rag: RagState = Field(default=RagState(), description="RAG state information")
    response: List[str] = Field(default_factory=list, description="Response from the system")
