import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from model import download_data, format_data, train_model, get_inference
from config import TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, TOKENS

app = FastAPI()

def update_data():
    """Download price data, format data, and train model."""
    try:
        for t in TOKENS.keys():
            TOKEN = t
            print(f"Updating data for {TOKEN}")

            files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
            print(f"Downloaded {len(files)} new files")
            format_data(files, DATA_PROVIDER, TOKEN)
            train_model(TIMEFRAME, TOKEN)

    except Exception as e:
        print(f"Failed to update data: {str(e)}")

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, update_data)

@app.get("/config")
def check_config():
    token_list = list(TOKENS.keys())
    return {"config": {"tokens": token_list, "timeframe": TIMEFRAME, "training_days": TRAINING_DAYS, "region": REGION, "data_provider": DATA_PROVIDER}}

@app.get("/inference/{token}")
def generate_inference(token: str):
    """Generate inference for a given token."""
    if not token or token.upper() not in TOKENS.keys():
        error_msg = "Token is required" if not token else "Token not supported"
        # Return an HTML response with the error message
        return HTMLResponse(content=error_msg, status_code=400)
    try:
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        # Return the inference result as HTML content
        return HTMLResponse(content=str(inference), status_code=200)
    except Exception as e:
        # Return an HTML response with the error message
        return HTMLResponse(content=str(e), status_code=500)

@app.get("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return HTMLResponse(content="0", status_code=200)
    except Exception:
        return HTMLResponse(content="1", status_code=500)
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return HTMLResponse(content=str(exc.detail), status_code=exc.status_code)

