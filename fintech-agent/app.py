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


class PredictiveMLTool:
    def __init__(self):
        self.model = joblib.load('loan_approval_model.pkl')
    
    def use(self, intent, age, income, loan_amount):
        input_data = pd.DataFrame(
            [[age, income, loan_amount, intent]],
            columns=['person_age', 'person_income', 'loan_amnt', 'loan_intent']
        )
        prob_approved = self.model.predict_proba(input_data)[0][0]
        print(f"Probability for the loan approval is: {prob_approved}")
        return "Eligible" if prob_approved > 0.5 else "Not Eligible"

class LoanAgent:
    def __init__(self):
        self.tools = [PredictiveMLTool()]
        self.llm = llm
    
    def process(self, query, params):
        intent = self.extract_intent(query)
        age = params.get('age')
        income = params.get('income')
        loan_amount = params.get('loan_amount')
        return self.tools[0].use(intent, age, income, loan_amount)
    
    def extract_intent(self, query):
        valid_intents = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
        prompt = f"Given the query: '{query}', classify the intent into one of: {', '.join(valid_intents)}. Respond with only the intent in uppercase (e.g., 'HOMEIMPROVEMENT'). If unsure, respond with 'PERSONAL'."
        response = self.llm(prompt)[0]['generated_text'].strip().upper()
        return response if response in valid_intents else 'PERSONAL'