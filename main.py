from fastapi import FastAPI # type: ignore
from pymongo.mongo_client import MongoClient # type: ignore
from pymongo.server_api import ServerApi # type: ignore
from dotenv import load_dotenv # type: ignore
import os

app = FastAPI()

load_dotenv()

user = os.getenv("MONGO_USER")
pwd = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
appname = os.getenv("MONGO_APPNAME")

uri = f"mongodb+srv://{user}:{pwd}@{cluster}/?appName={appname}"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

