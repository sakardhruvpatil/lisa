from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
import datetime

app = FastAPI()

client = MongoClient("mongodb://localhost:27017/")
db = client["app_database"]
collection = db["analytics_data"]

class AnalyticsData(BaseModel):
    accept: int
    reject: int
    date: str
    acceptance_rate: float

@app.post("/log-analytics-data/")
async def log_analytics(data: AnalyticsData):
    data_dict = data.dict()
    data_dict["date"] = datetime.datetime.strptime(data.date, "%Y-%m-%d %H:%M:%S")  # Convert to datetime object
    collection.insert_one(data_dict)
    return {"message": "Data logged successfully"}
