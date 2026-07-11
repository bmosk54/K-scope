"""Thin S3 helpers so downstream code can push/pull artifacts without repeating
boto3 boilerplate. Auth is the SageMaker execution role — no keys, no configure.
"""
import io
import os

import boto3
import numpy as np

from .. import config

_client = None


def s3():
    """Cached boto3 S3 client bound to the team bucket's region."""
    global _client
    if _client is None:
        _client = boto3.client("s3", region_name=config.REGION)
    return _client


def upload_file(local_path: str, key: str, bucket: str = config.BUCKET) -> str:
    """Upload a local file to s3://<bucket>/<key>. Returns the s3:// URI."""
    s3().upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def download_file(key: str, local_path: str, bucket: str = config.BUCKET) -> str:
    """Download s3://<bucket>/<key> to local_path. Returns local_path."""
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    s3().download_file(bucket, key, local_path)
    return local_path


def exists(key: str, bucket: str = config.BUCKET) -> bool:
    """True if the object exists."""
    from botocore.exceptions import ClientError
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def load_embeddings(model_key: str, split: str, bucket: str = config.BUCKET):
    """Pull an embeddings .npz straight into memory.

    Returns (feats: np.ndarray (N,dim), labels: np.ndarray (N,), class_names: list[str]).
    This is the function teammates call to run the causal battery on CPU.
    """
    key = config.embeddings_key(model_key, split)
    buf = io.BytesIO()
    s3().download_fileobj(bucket, key, buf)
    buf.seek(0)
    npz = np.load(buf, allow_pickle=True)
    return npz["feats"], npz["labels"], list(npz["class_names"])
