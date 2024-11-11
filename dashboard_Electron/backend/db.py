from pymongo import MongoClient

# Replace with your MongoDB URI or use localhost for local
client = MongoClient('mongodb://localhost:27017/')  # Default localhost connection
db = client['app_database"']  # MongoDB will create this database if it doesn't exist
collection = db['analytics_data']  # MongoDB will create this collection if it doesn't exist

# Example of inserting a document
data = {
    'day': '2024-11-11',
    'accept': 100,
    'reject': 5,
    'total': 105,
    'date': '2024-11-11T12:00:00'
}

# Insert a document into the collection
collection.insert_one(data)

print("Document inserted successfully!")
