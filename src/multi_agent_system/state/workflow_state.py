from pydantic import BaseModel, Field

class QueryEthics(BaseModel):
    is_ethical: bool = Field(default=True, description="Is the query ethical?")
    confidence: float = Field(default=1.0, description="Confidence level of the ethics evaluation")
    category: str = Field(default="", description="Category of ethical concern if any")
    reason: str = Field(default="", description="Reason if the query is not ethical")

class AppState(BaseModel):
    query: str = Field(default="", description="User query")
    ethics: QueryEthics = Field(default=QueryEthics(), description="Ethics check result")
    response: str = Field(default="", description="Response from the system")