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
model_file_path = os.path.join(data_base_path, "model.pkl")

# Set default values
default_data_provider = "binance"
default_timeframe = "1h"
default_model = "KernelRidge"

def get_model_file_path(token):
    return os.path.join(data_base_path, f"{token}_model.pkl")

def get_training_price_data_path(token):
    return os.path.join(data_base_path, f"{token}_price_data.csv")

class Token:
    def __init__(self, timeframe, training_days, region):
        self.timeframe = timeframe
        self.training_days = training_days
        self.region = region

TOKENS = {}
for key, value in inferences.items():
    TOKENS[key] = Token(**value)  # type: ignore

TOKEN = os.getenv("TOKEN")
if TOKEN is None:
    raise ValueError("Missing TOKEN variable")
else:
    TOKEN = TOKEN.upper()

MODEL_FILE_PATH = model_file_path


TRAINING_DAYS = os.getenv("TRAINING_DAYS")

# TIMEFRAME This should be in this form: 10min, 1h, 1d, 1m, etc.
TIMEFRAME = os.getenv("TIMEFRAME")
if not TIMEFRAME:
    TIMEFRAME = default_timeframe


MODEL = os.getenv("MODEL")
if MODEL not in ["LinearRegression", "SVR", "KernelRidge", "BayesianRidge"]:
    MODEL = default_model

REGION = os.getenv("REGION").lower()
if REGION in ["us", "com", "usa"]:
    REGION = "us"
else:
    REGION = "com"

DATA_PROVIDER = os.getenv("DATA_PROVIDER").lower()
if DATA_PROVIDER not in ["binance", "coingecko"]:
    DATA_PROVIDER = default_data_provider

CG_API_KEY = os.getenv("CG_API_KEY", default=None)
if CG_API_KEY is None:
    raise ValueError("Missing CG_API_KEY variable")
