# config/database.py
from pymongo import MongoClient  # type: ignore
from gridfs import GridFS        # type: ignore

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "inventory_db"

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# Collection for perfumes
perfumes_collection = db["perfumes"]

# GridFS bucket for storing images
fs = GridFS(db, collection="perfume_images")
