import requests
import time
import logging
from typing import Optional, Tuple, List, Any

from fastapi import APIRouter
from pmdarima_service import predict_auto_arima

# Configure logging
logger = logging.getLogger("arima_sidecar")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
logger.addHandler(handler)

def train_and_predict_arima(
    close_prices: List[float],
    actual_seq_length: int,
    market: str,
    ticker: str,
    bucket_name: str,
    model_override: bool = False,
    best_val_loss: Optional[float] = None,
    mode: str = "predict",
    retries: int = 3,
    backoff_factor: float = 1.5,
    timeout: int = 30,
    endpoint_url: str = "http://localhost:8000/predict_auto_arima"
) -> Tuple[Optional[Any], Optional[List[float]], Optional[float]]:
    """
    Call ARIMA sidecar FastAPI endpoint to train or predict.

    Returns:
    - model (None as placeholder in 3.11 context),
    - list of predictions,
    - validation loss (MSE)
    """

    payload = {
        "close_prices": close_prices,
        "actual_seq_length": actual_seq_length,
        "market": market,
        "ticker": ticker,
        "bucket_name": bucket_name,
        "model_override": model_override,
        "best_val_loss": best_val_loss,
        "mode": mode
    }

    logger.info(f"📨 Starting ARIMA sidecar request for {ticker} | mode={mode} | override={model_override}")

    attempt = 0
    while attempt < retries:
        try:
            logger.info(f"📡 Attempt {attempt + 1}/{retries} for {ticker}")
            response = requests.post(endpoint_url, json=payload, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            predictions = data.get("preds")
            val_loss = data.get("val_loss")

            if predictions is None or val_loss is None:
                logger.error(f"❌ Incomplete response received for {ticker}: {data}")
                break

            logger.info(f"✅ ARIMA sidecar success for {ticker} | val_loss={val_loss:.6f}")
            return None, predictions, val_loss

        except requests.Timeout:
            logger.warning(f"⏱️ Timeout on attempt {attempt + 1} for {ticker}")
        except requests.RequestException as re:
            logger.warning(f"⚠️ Request error on attempt {attempt + 1} for {ticker}: {re}")
        except ValueError as ve:
            logger.error(f"❌ Response validation error for {ticker}: {ve}")
            break
        except Exception as e:
            logger.exception(f"🔥 Unexpected exception during sidecar call for {ticker}: {e}")

        attempt += 1
        if attempt < retries:
            sleep_time = backoff_factor ** attempt
            logger.info(f"⏳ Retrying after {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    logger.error(f"❌ Failed to obtain ARIMA predictions for {ticker} after {retries} attempts")
    return None, None, None
