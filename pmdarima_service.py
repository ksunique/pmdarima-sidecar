# pmdarima_service.py

import logging
import warnings
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
from datetime import datetime
import pmdarima as pm  # type: ignore
from sklearn.metrics import mean_squared_error

from models.model_store import store_model_to_s3, load_model_from_s3
from store.postgres_store import PostgresWriter
from store.postgres_retrieve import PostgresRetriever

def train_auto_arima_model(close_prices, actual_seq_length, market, ticker, bucket_name):
    """
    Train or reuse an auto_arima model on closing prices.
    Returns:
    - arima_preds: forecast list
    """
    try:
        logger.info(f"🔧 Preparing data for auto_arima training for {ticker}...")

        close_prices = np.asarray(close_prices, dtype=np.float64)
        y_true = close_prices[actual_seq_length:]

        if len(close_prices) < 100:
            logger.error(f"❌ {ticker} - Insufficient data: only {len(close_prices)} rows. Skipping.")
            return []

        try:
            existing_model = load_model_from_s3(market, ticker, "arima", bucket_name, file_ext="pkl")
            if existing_model:
                logger.info(f"✅ Loaded cached ARIMA model for {ticker}")
                n_future = len(close_prices) - actual_seq_length
                forecast = existing_model.predict(n_periods=n_future)
                return forecast.tolist()
        except Exception as e:
            logger.warning(f"⚠️ Failed to use existing ARIMA model for {ticker}: {e}")

        logger.info(f"🧠 Training new ARIMA model for {ticker}...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            model = pm.auto_arima(
                close_prices,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=1,
                start_P=0, seasonal=False,
                d=1, D=1, trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

        n_future = len(close_prices) - actual_seq_length
        forecast = model.predict(n_periods=n_future)
        arima_preds = forecast.tolist()

        candidate_val_loss = mean_squared_error(y_true[-len(arima_preds):], arima_preds)
        logger.info(f"📈 Prediction completed for {ticker}. MSE: {candidate_val_loss:.6f}")

        retriever = PostgresRetriever()
        writer = PostgresWriter()
        best_val_loss = None

        meta_df = retriever.fetch_recent_model_metadata(market, ticker, "arima", limit=1)
        if not meta_df.empty:
            best_val_loss = meta_df.iloc[0]["val_loss"]
            logger.info(f"📄 Previous ARIMA MSE from metadata: {best_val_loss:.6f}")

        if best_val_loss is None or candidate_val_loss < best_val_loss:
            logger.info(f"📈 New ARIMA model outperforms previous. Saving to S3 and Postgres...")
            store_model_to_s3(model, market, ticker, "arima", bucket_name, file_ext="pkl")
            writer.write_model_metadata([{
                "market": market,
                "symbol": ticker,
                "model_type": "arima",
                "file_ext": "pkl",
                "val_loss": float(candidate_val_loss),
                "epochs_trained": 0,
                "created_at": datetime.utcnow()
            }])
        else:
            logger.info(f"⚖️ Existing model retained for {ticker} (val_loss={candidate_val_loss:.6f} >= best={best_val_loss:.6f})")

        return arima_preds

    except Exception as e:
        logger.exception(f"❌ Failed to train ARIMA model for {ticker}: {e}")
        return []
