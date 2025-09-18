import os
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings

load_dotenv()

def get_embedding_model():
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    return embedding_model
