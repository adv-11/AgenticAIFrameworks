import pandas as pd
import joblib
import pyarrow as pa
from sentence_transformers import SentenceTransformer
import lancedb
from mistralai import Mistral
import os
from abc import ABC, abstractmethod
from typing import Any


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


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

print (embedder)

# Connect to LanceDB
db = lancedb.connect('./lancedb_data')

loan_data = pd.read_csv('./credit_risk_dataset.csv')
print (loan_data.head())



# ML MODEL 

features = ['person_age', 'person_income', 'loan_amnt', 'loan_intent']
target = 'loan_status'

loan_data = loan_data.dropna(subset=features + [target])

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['loan_intent'])],
    remainder='passthrough'
)

X = loan_data[features]
y = loan_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")  # 0.84
joblib.dump(model, 'loan_approval_model.pkl')