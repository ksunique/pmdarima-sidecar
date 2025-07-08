import numpy as np
from pmdarima import auto_arima
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
import logging
import boto3
import joblib
import tempfile
import os
import botocore.exceptions
from model_store import store_model_to_s3, load_model_from_s3

print("NumPy version:", np.__version__)

# Initialize logger and app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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

# # -------------------------
# # S3 Utilities (localized)
# # -------------------------
# def load_model_from_s3(market, ticker, model_type, bucket_name, file_ext="pkl"):
#     key = f"market/{market}/{ticker}/{model_type}.{file_ext}"
#     try:
#         s3.download_file(bucket_name, key, "/tmp/model.pkl")
#         with open("/tmp/model.pkl", "rb") as f:
#             model = joblib.load(f)
#         return model
#     except botocore.exceptions.ClientError as e:
#         if e.response['Error']['Code'] == '404':
#             logger.info(f"📂 Model not found in S3: {key}")
#         else:
#             logger.error(f"❌ S3 error: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"❌ Failed to load model from S3: {e}")
#         return None

# def store_model_to_s3(model, market, ticker, model_type, bucket_name, file_ext="pkl"):
#     key = f"models/{market}/{ticker}/{model_type}.{file_ext}"
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
#         try:
#             joblib.dump(model, tmp_file.name)
#             s3.upload_file(tmp_file.name, bucket_name, key)
#             logger.info(f"✅ Stored model to S3: {key}")
#         except Exception as e:
#             logger.error(f"❌ Failed to upload model to S3 ({key}): {e}")
#         finally:
#             os.unlink(tmp_file.name)

# -------------------------
# Core Endpoint
# -------------------------
@app.post("/predict_auto_arima", response_model=ArimaResponse)
def predict_auto_arima(req: ArimaRequest):
    try:
        close_prices = np.asarray(req.close_prices, dtype=np.float64)
        y_true = close_prices[req.actual_seq_length:]

        # ✅ CORRECT
        model = load_model_from_s3(req.market, req.ticker, "auto-arima", req.bucket_name, file_ext="pkl") if req.model_override else None

        if model is None:
            model = auto_arima(
                close_prices,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            logger.info(f"✅ Trained new ARIMA model for {req.ticker}")
        else:
            logger.info(f"➡️ Using existing ARIMA model for prediction: {req.ticker}")

        try:
            preds = model.predict(n_periods=len(y_true))
            val_loss = float(np.mean((y_true - preds) ** 2)) if len(y_true) == len(preds) else None
            logger.info(f"📈 Prediction complete. MSE: {val_loss:.6f}")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise HTTPException(status_code=500, detail="ARIMA prediction failed")

        # Save model if mode=train and it's better
        if req.mode == "train" and (req.best_val_loss is None or (val_loss is not None and val_loss < req.best_val_loss)):
            store_model_to_s3(model, req.market, req.ticker, "auto-arima", req.bucket_name, file_ext = "pkl")

        return ArimaResponse(
            model_exists=(model is not None),
            preds=preds.tolist(),
            val_loss=val_loss,
            status="success"
        )

    except Exception as ex:
        logger.exception(f"Unhandled exception in ARIMA endpoint: {ex}")
        raise HTTPException(status_code=500, detail="Unhandled exception during ARIMA processing")
