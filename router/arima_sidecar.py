import requests
import time
import logging
from typing import Optional, Tuple, List, Any

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
    endpoint_url: str = "http://localhost:8000/predict-auto-arima"
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

    attempt = 0
    while attempt < retries:
        try:
            logger.info(f"📡 Sidecar call attempt {attempt + 1} for {ticker} | mode={mode}")
            response = requests.post(endpoint_url, json=payload, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            predictions = data.get("predictions")
            val_loss = data.get("val_loss")

            if predictions is None or val_loss is None:
                raise ValueError(f"Sidecar returned incomplete data: {data}")

            logger.info(f"✅ Sidecar successful for {ticker} | val_loss={val_loss:.6f}")
            return None, predictions, val_loss

        except requests.RequestException as re:
            logger.warning(f"⚠️ Sidecar request error on attempt {attempt + 1}: {re}")
        except ValueError as ve:
            logger.error(f"❌ Invalid response: {ve}")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error on attempt {attempt + 1}: {e}")

        attempt += 1
        if attempt < retries:
            sleep_time = backoff_factor ** attempt
            logger.info(f"⏳ Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    logger.error(f"❌ Failed to get ARIMA predictions after {retries} attempts for {ticker}")
    return None, None, None
