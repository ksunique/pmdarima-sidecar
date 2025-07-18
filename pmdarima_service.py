# pmdarima_service.py

import logging
import warnings
import sys
import os
logger = logging.getLogger("arima-sidecar")  # Use consistent logger name
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(handler)  # Add formatter for consistency

import numpy as np
from pmdarima import auto_arima
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import boto3
import joblib
import tempfile
from statsmodels.tsa.stattools import adfuller

try:
    from model_store import store_model_to_s3, load_model_from_s3
except ModuleNotFoundError as e:
    logger.error(f"❌ Failed to import model_store: {e}")
    logger.info(f"Current PYTHONPATH: {sys.path}")
    logger.info(f"Current directory contents: {os.listdir('/app')}")
    raise

logger.info("✅ pmdarima_service.py loaded. NumPy version: %s", np.__version__)

app = FastAPI()
s3 = boto3.client("s3")

class ArimaRequest(BaseModel):
    close_prices: List[float]
    actual_seq_length: int
    market: str
    ticker: str
    bucket_name: str
    model_override: bool = False
    best_val_loss: Optional[float] = None
    mode: str

class ArimaResponse(BaseModel):
    model_exists: bool
    preds: List[float]
    val_loss: Optional[float] = None
    status: str

@app.post("/predict_auto_arima", response_model=ArimaResponse)
def predict_auto_arima(req: ArimaRequest):
    logger.info(f"🚀 Received ARIMA request for {req.ticker} (mode={req.mode}, override={req.model_override})")
    logger.debug(f"Request details: market={req.market}, bucket={req.bucket_name}, seq_length={req.actual_seq_length}")

    try:
        close_prices = np.asarray(req.close_prices, dtype=np.float64)
        logger.info(f"📊 ARIMA received {len(close_prices)} close prices for {req.ticker}")
        y_true = close_prices[req.actual_seq_length:]

        # Stationarity check
        result = adfuller(close_prices)
        if result[1] > 0.05:
            logger.warning(f"⚠️ Non-stationary data detected for {req.ticker} (p-value={result[1]:.4f}). Applying differencing.")
            close_prices = np.diff(close_prices)
            y_true = y_true[1:]
        else:
            logger.info(f"✅ Stationary data confirmed for {req.ticker} (p-value={result[1]:.4f}).")

        model = None
        if req.model_override:
            logger.info(f"🔄 Model override enabled. Attempting to load existing ARIMA model for {req.ticker}")
            model = load_model_from_s3(req.market, req.ticker, "auto-arima", req.bucket_name, file_ext="pkl")

        if model is None:
            logger.info(f"🧠 Training new ARIMA model for {req.ticker}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                model = auto_arima(
                    close_prices,
                    seasonal=True,
                    m=5,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore"
                )
            logger.info(f"✅ Successfully trained new ARIMA model for {req.ticker}. Model summary: {model.summary()}")
        else:
            logger.info(f"📦 Loaded existing ARIMA model for {req.ticker}")

        try:
            preds = model.predict(n_periods=len(y_true))
            if len(preds) != len(y_true):
                logger.error(f"❌ Prediction length mismatch for {req.ticker}: preds={len(preds)}, y_true={len(y_true)}")
                raise ValueError("Prediction and true value lengths do not match")
            if np.any(np.isnan(preds)):
                logger.error(f"❌ NaN values in predictions for {req.ticker}. Predictions: {preds}")
                raise ValueError("NaN values detected in predictions")
            val_loss = float(np.mean((y_true - preds) ** 2))
            logger.info(f"📈 Prediction completed for {req.ticker}. MSE: {val_loss:.6f}")
            logger.debug(f"📊 ARIMA predictions preview: {preds[:5]} | True values preview: {y_true[:5]}")
        except Exception as e:
            logger.exception(f"❌ Prediction failed for {req.ticker}: {e}")
            raise HTTPException(status_code=500, detail="ARIMA prediction failed")

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
