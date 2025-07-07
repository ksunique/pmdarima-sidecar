from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np
import joblib
import tempfile

from pmdarima import auto_arima

from utils.s3_utils import load_model_from_s3, store_model_to_s3

app = FastAPI()

# ----------- Request / Response Schemas -----------

class ARIMARequest(BaseModel):
    close_prices: List[float]
    actual_seq_length: int
    market: str
    ticker: str
    bucket_name: str
    mode: str  # 'predict' or 'train'
    best_val_loss: Optional[float] = None
    model_override: Optional[bool] = False


class ARIMAResponse(BaseModel):
    predictions: Optional[List[float]]
    val_loss: Optional[float]
    message: str


# ----------- FastAPI Endpoint -----------

@app.post("/predict-auto-arima", response_model=ARIMAResponse)
def predict_auto_arima(req: ARIMARequest):
    try:
        close_prices = np.asarray(req.close_prices, dtype=np.float64)
        y_true = close_prices[req.actual_seq_length:]

        model = None

        if req.mode == "predict":
            model = load_model_from_s3(req.market, req.ticker, "arima", req.bucket_name, file_ext="pkl")
            if model is None:
                raise HTTPException(status_code=404, detail="ARIMA model not found in S3 for prediction.")
        else:  # train
            model = auto_arima(
                close_prices,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )

        preds = model.predict(n_periods=len(y_true)).tolist()
        val_loss = float(np.mean((y_true - np.array(preds)) ** 2))

        # Save if training and either no baseline or better val_loss
        if req.mode == "train" and (not req.model_override) and (req.best_val_loss is None or val_loss < req.best_val_loss):
            store_model_to_s3(model, req.market, req.ticker, "arima", req.bucket_name, file_ext="pkl")

        return ARIMAResponse(
            predictions=preds,
            val_loss=val_loss,
            message="✅ ARIMA prediction successful."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ARIMA error: {str(e)}")
