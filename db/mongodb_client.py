from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
import os
import urllib.parse
import streamlit as st

# === Load credentials securely ===
load_dotenv()  # loads .env from project root by default

user = urllib.parse.quote_plus(st.secrets["MONGO_USER"])
password = urllib.parse.quote_plus(st.secrets["MONGO_PASSWORD"])
MONGO_CLUSTER = st.secrets["MONGO_CLUSTER"]
MONGO_DB = st.secrets["MONGO_DB"]
APP_NAME = st.secrets["APP_NAME"]

# === Build the connection URI ===
MONGODB_URI = (
    f"mongodb+srv://{user}:{password}@{MONGO_CLUSTER}.mongodb.net/"
    f"{MONGO_DB}?retryWrites=true&w=majority&appName={APP_NAME}"
)
#uri = 'mongodb+srv://' + username + ':' + password + "@diamind.q4fmjuw.mongodb.net/?retryWrites=true&w=majority&appName=DiaMind"

# === Create a MongoDB client ===
try:
    mongodb_client = MongoClient(
        MONGODB_URI,
        server_api=ServerApi('1')
    )
    mongodb_client.admin.command("ping")
    print("✅ MongoDB connection established.")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")
    mongodb_client = None
