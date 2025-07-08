# model_store.py

import boto3
import joblib
import os
import tempfile
import numpy as np
# from tensorflow.keras.models import load_model  # type: ignore
# from tensorflow.keras.saving import load_model as load_keras_model  # for .keras support # type: ignore


# def write_model_to_temp_file(model, file_ext: str):
#     """
#     Serializes a model to a temporary file and returns the file path.

#     Parameters:
#     - model: the model object to save
#     - file_ext: str, file extension (e.g., 'keras', 'h5', 'pkl')

#     Returns:
#     - temp_file_path: str, path to the temporary file
#     """
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
#         if file_ext in ["keras", "h5"]:
#             model.save(tmp_file.name)
#         else:
#             joblib.dump(model, tmp_file.name)
#         tmp_file.flush()
#         return tmp_file.name

def load_model_from_s3(market, ticker, model_type, bucket_name, file_ext="pkl"):
    key = f"market/{market}/{ticker}/{model_type}.{file_ext}"
    try:
        s3.download_file(bucket_name, key, "/tmp/model.pkl")
        with open("/tmp/model.pkl", "rb") as f:
            model = joblib.load(f)
        return model
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"📂 Model not found in S3: {key}")
        else:
            logger.error(f"❌ S3 error: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load model from S3: {e}")
        return None

def store_model_to_s3(model, market, ticker, model_type, bucket_name, file_ext="pkl"):
    key = f"models/{market}/{ticker}/{model_type}.{file_ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        try:
            joblib.dump(model, tmp_file.name)
            s3.upload_file(tmp_file.name, bucket_name, key)
            logger.info(f"✅ Stored model to S3: {key}")
        except Exception as e:
            logger.error(f"❌ Failed to upload model to S3 ({key}): {e}")
        finally:
            os.unlink(tmp_file.name)

# def store_model_to_s3(model, market: str, ticker: str, model_type: str, bucket_name: str, file_ext: str = "keras"):
#     """
#     Stores the trained model to S3 at path: {market}/{model_type}/{ticker}.{ext}
#     Example: s3://ml-trained-models/NSE/lstm/RELIANCE.keras
#     Assumes S3 versioning is enabled.
#     """
#     s3 = boto3.client("s3")
#     s3_path = f"{market}/{model_type}/{ticker}.{file_ext}"

#     temp_file_path = write_model_to_temp_file(model, file_ext)
#     s3.upload_file(temp_file_path, bucket_name, s3_path)

#     print(f"✅ Stored {model_type.upper()} model for {ticker} at s3://{bucket_name}/{s3_path}")


# def load_model_from_s3(market: str, ticker: str, model_type: str, bucket_name: str, file_ext: str = "keras"):
#     """
#     Loads a model from S3 if it exists.

#     Parameters:
#     - market: str
#     - ticker: str
#     - model_type: str (e.g., 'nn', 'lstm')
#     - bucket_name: str
#     - file_ext: str, file extension (default: 'keras')

#     Returns:
#     - model if found, else None
#     """
#     s3 = boto3.client("s3")
#     s3_path = f"{market}/{model_type}/{ticker}.{file_ext}"

#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
#             s3.download_file(bucket_name, s3_path, tmp_file.name)
#             if file_ext == "keras":
#                 model = load_keras_model(tmp_file.name)
#             elif file_ext == "h5":
#                 model = load_model(tmp_file.name)
#             else:
#                 model = joblib.load(tmp_file.name)
#             print(f"✅ Loaded existing {model_type.upper()} model for {ticker} from s3://{bucket_name}/{s3_path}")
#             return model
#     except s3.exceptions.ClientError:
#         print(f"ℹ️ No existing model found for {ticker} at s3://{bucket_name}/{s3_path}")
#         return None
