from pymongo import MongoClient # type: ignore
from dotenv import load_dotenv # type: ignore
import os

load_dotenv()

user = os.getenv("MONGO_USER")
pwd = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
appname = os.getenv("MONGO_APPNAME")

uri = f"mongodb+srv://{user}:{pwd}@{cluster}/?appName={appname}"
client = MongoClient(uri)
db = client.InventoryDB

collection_name = db["Perfume_collection"]