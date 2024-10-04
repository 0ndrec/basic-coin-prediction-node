import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
import threading
from contextlib import asynccontextmanager
from model import download_data, format_data, train_model, get_inference
from config import DATA_PROVIDER, TOKENS



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

def download_train(TOKEN, DATA_PROVIDER):
    TRAINING_DAYS = TOKENS[TOKEN].training_days
    REGION = TOKENS[TOKEN].region
    TIMEFRAME = TOKENS[TOKEN].timeframe

    files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files, DATA_PROVIDER, TOKEN)
    train_model(TIMEFRAME, TOKEN)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    loop = asyncio.get_event_loop()
    # Startup: Run update_data asynchronously
    await loop.run_in_executor(None, update_data)
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/config")
def check_config():
    return TOKENS

@app.get("/inference/{token}")
async def generate_inference(token: str):
    TIMEFRAME = TOKENS[token.upper()].timeframe
    REGION = TOKENS[token.upper()].region

    if not token or token.upper() not in TOKENS.keys():
        error_msg = "Token is required" if not token else "Token not supported"
        return HTMLResponse(content=error_msg, status_code=400)
    try:
        inference = await asyncio.to_thread(get_inference, token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return HTMLResponse(content=str(inference), status_code=200)
    except Exception as e:
        return HTMLResponse(content=str(e), status_code=500)

@app.get("/update")
async def update():
    """Update data and return status."""
    try:
        # Run update_data asynchronously
        await asyncio.to_thread(update_data)
        return HTMLResponse(content="0", status_code=200)
    except Exception:
        return HTMLResponse(content="1", status_code=500)
    
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return HTMLResponse(content=str(exc.detail), status_code=exc.status_code)

