import os
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from chromadb.utils import embedding_functions


## Dúvida
# Devo usar apenas o valor definido no config, ou deixo livre
class ModelFactory:

    @staticmethod
    def get_llm(
        provider: str,
        model_name: str,
        temperature: float,
    ) -> ChatOpenAI:
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        # elif provider == "ollama":
        else:
            raise ValueError(f"Provedor de LLM '{provider}' não suportado.")

    @staticmethod
    def get_embedding_function(
        provider: str,
        model_name: str,
    ) -> Any:
        if provider == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=model_name
            )
        
        elif provider == "huggingface":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        
        else:
            raise ValueError(f"Provedor de embedding '{provider}' não suportado.")