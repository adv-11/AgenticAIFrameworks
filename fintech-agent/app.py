import pandas as pd
import joblib
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import lancedb
from mistralai import Mistral
import os
from abc import ABC, abstractmethod
from typing import Any

# Abstract Tool Class
class Tool(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def use(self, *args, **kwargs) -> Any:
        pass

# Initialize Mistral and Embedder
api_key = os.environ.get("MISTRAL_API_KEY", "xxxxxxxxxxxxx")  
if not api_key:
    raise ValueError("Please set the MISTRAL_API_KEY environment variable.")

model = "mistral-large-latest"
client = Mistral(api_key=api_key)
hf_token = "hf_xxxxxxx"
embedder = SentenceTransformer('all-MiniLM-L6-v2', token=hf_token)

# Connect to LanceDB
db = lancedb.connect('./lancedb_data')