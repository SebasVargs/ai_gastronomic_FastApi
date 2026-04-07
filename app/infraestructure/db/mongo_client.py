import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://admin:admin1234567890@cluster01.ycsvthv.mongodb.net/restaurantesDB?retryWrites=true&w=majority&appName=Cluster01")
DB_NAME = "restaurantesDB"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def get_db():
    return db