from pymongo import MongoClient

# Replace with your MongoDB URI or use localhost for local
client = MongoClient('mongodb://localhost:27017/')
db = client['app_database']  # Use your preferred database name
collection = db['analytics_data']  # Collection where data will be stored