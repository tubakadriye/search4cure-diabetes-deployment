from google.cloud import storage
import os
import streamlit as st

# Set GCS project and bucket
GCS_PROJECT = st.secrets["GCS_PROJECT"]
GCS_BUCKET = st.secrets["GCS_BUCKET"]

# Initialize your GCS client and bucket
gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)

def upload_image_to_gcs(key: str, data: bytes) -> None:
    """
    Upload image to GCS.

    Args:
        key (str): Unique identifier for the image in the bucket.
        data (bytes): Image bytes to upload.
    """
    blob = gcs_bucket.blob(key)
    blob.upload_from_string(data, content_type="image/png")

#set environment variables before running:
#export GCS_PROJECT="your-project-id"
#export GCS_BUCKET="your-bucket-name"
#export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"

from io import BytesIO
from google.cloud import storage

def get_image_from_gcs(gcs_bucket, key: str) -> bytes:
    """
    Download image bytes from GCS.

    Args:
        gcs_bucket: GCS bucket instance.
        key (str): Blob key in the bucket.

    Returns:
        bytes: Image bytes.
    """
    blob = gcs_bucket.blob(key)
    return blob.download_as_bytes()