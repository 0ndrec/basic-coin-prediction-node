import json
import pickle
import pathlib
import pandas as pd
import numpy as np
from typing import List
from datetime import datetime
from flask import Flask, jsonify, Response, request
from model import download_data, format_data, train_model
from config import model_file_path, data_base_path, TOKENS

app = Flask(__name__)



def update_data(token, years: List[str], months: List[str]):
    """Download price data, format data and train model."""
    "Format token add stablecoin to string"
    token = token.upper()
    pair = f"{token}USDT"
    download_data(symbols=[pair], years=years, months=months)
    format_data(token) #lowwer case str token
    train_model(token) #lower case str token


def get_inference(token: str = "ETH"):
    """Load model with token name and predict current price."""
    model_ = model_file_path.replace(".pkl", f"_{token.lower()}.pkl")

    if pathlib.Path(model_).exists():
        with open(model_, "rb") as f:
            loaded_model = pickle.load(f)
    else:
        # Upadate model
        return f"Use POST request to update data. Example: /update/{token}/from_year/to_year/from_month/to_month. For show loaded files use /datafiles"

    now_timestamp = pd.Timestamp(datetime.now()).timestamp()
    X_new = np.array([now_timestamp]).reshape(-1, 1)
    current_price_pred = loaded_model.predict(X_new)

    return current_price_pred[0][0]


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token not in TOKENS:
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_inference(token)
        return Response(str(inference), status=200)
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data(token="ETH", years=["2020", "2021", "2022", "2023", "2024"], months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
        return "0"
    except Exception:
        return "1"
    
@app.route("/update/<string:token>/<int:from_years>/<int:to_years>/<int:from_months>/<int:to_months>")
def update_tokens(token, from_years, to_years, from_months, to_months):
    # Parse the range of years and months. Generate a list of years and months
    years_list = list(range(from_years, to_years + 1))
    months_list = list(range(from_months, to_months + 1))
    """Update data and return status."""
    try:
        update_data(token=token, years=years_list, months=months_list)
        return "0"
    except Exception as e:
        # Return example jsonify
        return jsonify({"error": f"{e}", "Example": "/update/{token}/from_year/to_year/from_month/to_month. For show loaded files use /datafiles"})
    
@app.route("/datafiles")
def files():
    """Return a list of downloaded files in data_base_path with timestamp and size.

    Returns:
        JSON: JSON object with data_files and model_files lists.
    """
    try:
        # Show size in KB
        files_dict = {}
        _files = [f.name for f in pathlib.Path(data_base_path).iterdir() if f.is_file()]
        for f in _files:
            files_dict[f] = pathlib.Path(data_base_path, f).stat().st_size / 1024
        

        return jsonify(files_dict)
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route("/datafiles/binance")
def binance_files():
    """Return a list of downloaded files in data_base_path with timestamp and size.

    Returns:
        JSON: JSON object with data_files and model_files lists.
    """
    try:
        # Show size in KB
        files_dict = {}
        # edd to path end binance dir. use pathlib
        binance_dir = pathlib.Path(data_base_path, "binance")
        klines = pathlib.Path(binance_dir, "futures-klines")
        print(binance_dir)
        _files = [f.name for f in pathlib.Path(klines).iterdir() if f.is_file()]
        for f in _files:
            files_dict[f] = pathlib.Path(klines, f).stat().st_size / 1024
        

        return jsonify(files_dict)
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/')
def index():
    current_url = request.url
    return f'The current URL is: {current_url}'


if __name__ == "__main__":
    _files = [f.name for f in pathlib.Path(data_base_path).iterdir() if f.is_file()]
    if len(_files) == 0:
        update_data(token="ETH", years=["2020", "2021", "2022", "2023", "2024"], months=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
    app.run(host="0.0.0.0", port=8000)
