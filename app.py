import json
from flask import Flask, Response
from pathlib import Path
import asyncio
import threading
from model import download_data, format_data, train_model, get_inference
from config import DATA_PROVIDER, TOKENS, data_base_path



app = Flask(__name__)



def download_train(token, DATA_PROVIDER):
    TRAINING_DAYS = TOKENS[token].training_days
    REGION = TOKENS[token].region
    TIMEFRAME = TOKENS[token].timeframe

    files = download_data(token, TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files, DATA_PROVIDER, token)
    train_model(TIMEFRAME, token)


def update_data():
    threads = []
    try:
        for TOKEN in TOKENS.keys():
            print(f"Updating data for {TOKEN}")
            thread = threading.Thread(target=download_train, args=(TOKEN, DATA_PROVIDER))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except Exception as e:
        print(f"Failed to update data: {str(e)}")


@app.route("/tokens")
async def check_tokens():
    tokens = {}
    names = list(TOKENS.keys())
    for name in names:
        tokens[name] = {
            "timeframe": TOKENS[name].timeframe,
            "training_days": TOKENS[name].training_days
        }
    return tokens

@app.route("/models")
async def check_models():
    models = Path(data_base_path).glob("*.pkl")
    return [str(model) for model in models]
    

@app.route("/update")
async def update():
    try:
        await asyncio.to_thread(update_data)
        return "0"
    except Exception:
        return "1"
    

@app.route("/inference/<string:token>")
async def generate_inference(token: str):
    TIMEFRAME = TOKENS[token.upper()].timeframe
    REGION = TOKENS[token.upper()].region

    if not token or token.upper() not in TOKENS.keys():
        error_msg = "Token is required" if not token else "Token not supported"
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')
    try:
        inference = await asyncio.to_thread(get_inference, token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(str(inference), status=200)
        
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    

if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8000)