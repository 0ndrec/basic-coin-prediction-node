import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print("Failed to load .env file")

# Read tokens.json to variable inferences
with open("tokens.json", "r") as f:
    inferences = json.load(f)


app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
MODEL_FILE_PATH = os.path.join(data_base_path, "model.pkl")



def get_model_file_path(token):
    return os.path.join(data_base_path, f"{token}_model.pkl")

def get_training_price_data_path(token):
    return os.path.join(data_base_path, f"{token}_price_data.csv")

class Token:
    def __init__(self, timeframe, training_days, region):
        self.timeframe = timeframe
        self.training_days = training_days
        if region in ["us", "com", "usa"]:
            self.region = "us"
        else:
            self.region = "com"
    def __repr__(self):
        return f"Token(timeframe={self.timeframe}, training_days={self.training_days}, region={self.region})"

TOKENS = {}
for key, value in inferences.items():
    TOKENS[key] = Token(**value)  # type: ignore



#________________DATA FROM ENV VAR__________________
MODEL = os.getenv("MODEL")
if MODEL not in ["LinearRegression", "SVR", "KernelRidge", "BayesianRidge"]:
    # Set default model
    MODEL = "KernelRidge"

DATA_PROVIDER = os.getenv("DATA_PROVIDER").lower()
if DATA_PROVIDER not in ["binance", "coingecko"]:
    # Set default data provider
    DATA_PROVIDER = "binance"

CG_API_KEY = os.getenv("CG_API_KEY", default=None)
if CG_API_KEY is None:
    raise ValueError("Missing CG_API_KEY variable")
