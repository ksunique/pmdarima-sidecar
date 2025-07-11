import logging
import numpy as np
from pmdarima import auto_arima
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import boto3
import joblib
import tempfile
import os

from model_store import store_model_to_s3, load_model_from_s3

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("✅ pmdarima_service.py loaded. NumPy version: %s", np.__version__)

# FastAPI app and boto3 client
app = FastAPI()
s3 = boto3.client("s3")

# Request and Response Schemas
class ArimaRequest(BaseModel):
    close_prices: List[float]
    actual_seq_length: int
    market: str
    ticker: str
    bucket_name: str
    model_override: bool = False
    best_val_loss: Optional[float] = None
    mode: str  # 'train' or 'predict'

class ArimaResponse(BaseModel):
    model_exists: bool
    preds: List[float]
    val_loss: Optional[float] = None
    status: str

# Core Endpoint
@app.post("/predict_auto_arima", response_model=ArimaResponse)
def predict_auto_arima(req: ArimaRequest):
    logger.info(f"🚀 Received ARIMA request for {req.ticker} (mode={req.mode}, override={req.model_override})")

    try:
        close_prices = np.asarray(req.close_prices, dtype=np.float64)
        logger.info(f"📊 ARIMA received {len(close_prices)} close prices for {req.ticker}")
        y_true = close_prices[req.actual_seq_length:]

        # Load model if override enabled
        model = None
        if req.model_override:
            logger.info(f"🔄 Model override enabled. Attempting to load existing ARIMA model for {req.ticker}")
            model = load_model_from_s3(req.market, req.ticker, "auto-arima", req.bucket_name, file_ext="pkl")

        if model is None:
            logger.info(f"🧠 Training new ARIMA model for {req.ticker}")
            model = auto_arima(
                close_prices,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            logger.info(f"✅ Successfully trained new ARIMA model for {req.ticker}")
        else:
            logger.info(f"📦 Loaded existing ARIMA model for {req.ticker}")

        try:
            preds = model.predict(n_periods=len(y_true))
            if len(preds) != len(y_true):
                raise ValueError("Prediction and true value lengths do not match")
            val_loss = float(np.mean((y_true - preds) ** 2))
            logger.info(f"📈 Prediction completed for {req.ticker}. MSE: {val_loss:.6f}")
        except Exception as e:
            logger.error(f"❌ Prediction failed for {req.ticker}: {e}")
            raise HTTPException(status_code=500, detail="ARIMA prediction failed")

        # Save model conditionally
        if req.mode == "train":
            if req.best_val_loss is None:
                logger.info(f"📤 Saving model for {req.ticker} (no prior val_loss available)")
                store_model_to_s3(model, req.market, req.ticker, "auto-arima", req.bucket_name, file_ext="pkl")
            elif val_loss is not None and val_loss < req.best_val_loss:
                logger.info(f"📤 Saving improved model for {req.ticker} (new val_loss={val_loss:.6f} < best={req.best_val_loss:.6f})")
                store_model_to_s3(model, req.market, req.ticker, "auto-arima", req.bucket_name, file_ext="pkl")
            else:
                logger.info(f"⚖️ Existing model retained for {req.ticker} (val_loss={val_loss:.6f} >= best={req.best_val_loss})")

        return ArimaResponse(
            model_exists=True,
            preds=preds.tolist(),
            val_loss=val_loss,
            status="success"
        )

    except Exception as ex:
        logger.exception(f"❌ Unhandled exception in ARIMA endpoint for {req.ticker}: {ex}")
        raise HTTPException(status_code=500, detail="Unhandled exception during ARIMA processing")
