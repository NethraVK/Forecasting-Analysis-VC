from pymongo import MongoClient
import pandas as pd

# Connect to your DB
client = MongoClient("mongodb://localhost:27017/")
db = client["event_hosting_platform"]
events = list(db["Event"].find({}))
# Pull all documents
events = list(db["Event"].find({}))

# Convert to DataFrame
df = pd.DataFrame(events)

pd.set_option("display.max_rows", None)     # show rows
pd.set_option("display.max_columns", None)  # show columns
print(df)
 