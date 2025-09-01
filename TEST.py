from pymongo import MongoClient
from datetime import datetime, timedelta
import random
# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["event_hosting_platform"]
# 2. Drop collections if they exist
collections = ["Event", "EventHostUser", "EventInvite", "Ratings"]
for col in collections:
    db[col].drop()
# 3. Mock Data Generation

industries = ["Tech", "Fashion", "Automotive", "Medical"]
languages = ["English", "Arabic", "Mandarin"]
skills = ["Customer Service", "Sales", "Event Coordination"]

# ---------- Event Collection ----------
event_docs = []
for i in range(1, 11):  # 10 events
    event_date = datetime(2025, random.randint(1, 12), random.randint(1, 28))
    event_docs.append({
        "_id": f"E{i}",
        "name": f"{random.choice(industries)} Expo {i}",
        "industry": random.choice(industries),
        "location": f"Venue {random.randint(1, 5)}",
        "startDate": event_date,
        "endDate": event_date + timedelta(days=random.randint(1, 3)),
        "description": f"Description for event {i}"
    })
db["Event"].insert_many(event_docs)

# ---------- EventHostUser Collection ----------
host_docs = []
for i in range(1, 16):  # 15 hosts
    available_dates = []
    # Each host available 3 random dates
    for _ in range(3):
        d = datetime(2025, random.randint(1, 12), random.randint(1, 28))
        available_dates.append(d.strftime("%Y-%m-%d"))

    host_docs.append({
        "_id": f"H{i}",
        "fullName": f"Host {i}",
        "languages": random.sample(languages, k=random.randint(1, 2)),
        "skills": random.sample(skills, k=random.randint(1, 2)),
        "industries": random.sample(industries, k=random.randint(1, 2)),
        "yearsOfExperience": random.randint(1, 5),
        "availabilityDates": available_dates
    })
db["EventHostUser"].insert_many(host_docs)

# ---------- EventInvite Collection ----------
invite_docs = []
for i in range(1, 11):  # each event has 1-3 exhibitors requesting hosts
    for j in range(random.randint(1, 3)):
        invite_docs.append({
            "_id": f"INV{i}-{j + 1}",
            "eventID": f"E{i}",
            "exhibitorName": f"Exhibitor {i}-{j + 1}",
            "hostessRequirements": random.randint(1, 5),  # number of hosts requested
            "status": random.choice(["In Progress", "Accepted", "Declined"])
        })
db["EventInvite"].insert_many(invite_docs)

# ---------- Ratings Collection ----------
ratings_docs = []
for host in host_docs:
    for _ in range(random.randint(1, 4)):  # each host rated 1-3 times
        ratings_docs.append({
            "hostID": host["_id"],
            "eventID": f"E{random.randint(1, 10)}",
            "rating": random.randint(1, 5)
        })
db["Ratings"].insert_many(ratings_docs)
print("Mock data generated for MongoDB!")
