from fastapi import FastAPI
from pydantic import BaseModel
from db import collection

app = FastAPI()

class SingleDataEntry(BaseModel):
    type: str  # Type can be "accepted", "rejected", or "total"
    value: int

@app.post("/log-analytics-data/")
async def add_entry(data: SingleDataEntry):
    # Increment the specific count in MongoDB
    if data.type in ["accepted", "rejected", "total"]:
        collection.update_one(
            {"_id": "countData"},
            {"$inc": {data.type: data.value}},  # Increment the specific field
            upsert=True
        )
        # Fetch updated data to return as a response
        updated_data = collection.find_one({"_id": "countData"})
        return {"message": "Entry added successfully", "data": updated_data}
    return {"message": "Invalid entry type"}
