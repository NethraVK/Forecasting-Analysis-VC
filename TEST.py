from pymongo import MongoClient
from datetime import datetime, timedelta
import random

# 1. Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["event_hosting_platform"]

# 2. Drop collections if they exist
collections = [
    "Event", "EventHostUser", "EventHostProfile", "ExhibitorUser", "ExhibitorProfile",
    "EventInvite", "ActivityLog", "SupportRequest", "UnavailableDate", "Ratings"
]
for col in collections:
    db[col].drop()

# 3. Mock Data Generation

industries = ["Tech", "Fashion", "Automotive", "Medical"]
languages = ["English", "Arabic", "Mandarin"]
event_types = ["Expo", "Conference", "Summit", "Show"]
certifications = ["First Aid", "Hospitality", "Sales Pro", "Event Mgmt"]
locations = [f"Venue {i}" for i in range(1, 6)]

def rand_phone():
    return "+971" + str(random.randint(500000000, 599999999))

# ---------- ExhibitorUser Collection ----------
exhibitors = []
for i in range(1, 81):  # 80 exhibitors across industries
    industry = random.choice(industries)
    exhibitors.append({
        "_id": f"X{i}",
        "companyName": f"Company {i}",
        "website": f"https://company{i}.example.com",
        "industry": industry,
        "companyBio": f"Bio for company {i}",
        "keyProducts": [f"Product {i}-A", f"Product {i}-B"],
        "contactName": f"Contact {i}",
        "jobTitle": "Manager",
        "contactEmail": f"contact{i}@company.com",
        "phone": rand_phone(),
        "password": "hashed_pw",
        "representativePhoto": None,
        "businessCard": None
    })
db["ExhibitorUser"].insert_many(exhibitors)

# ---------- Event Collection ----------
events = []
today = datetime.today().replace(day=1)
# generate ~12 months of events ending this month (10 events per month)
event_id = 1
for m_offset in range(12, -1, -1):  # past 12 months to current
    month_dt = (today - timedelta(days=30*m_offset)).replace(day=1)
    for _ in range(10):
        start_day = random.randint(1, 25)
        start = month_dt.replace(day=min(start_day, 28))
        end = start + timedelta(days=random.randint(1, 3))
        # derive status relative to today
        if end < datetime.today():
            status = "Ended"
        elif start > datetime.today():
            status = "Upcoming"
        else:
            status = "Ongoing"
        events.append({
            "_id": f"E{event_id}",
            "name": f"Event {event_id}",
            "dates": {"start": start, "end": end},
            "location": random.choice(locations),
            "description": f"Description for event {event_id}",
            "status": status,
            "participants": []
        })
        event_id += 1
db["Event"].insert_many(events)

# ---------- ExhibitorProfile Collection ----------
exhibitor_profiles = []
xp_id = 1
for ev in events:
    # 1-4 exhibitors per event
    for _ in range(random.randint(1, 4)):
        exhibitor = random.choice(exhibitors)
        event_date = ev["dates"]["start"].strftime("%Y-%m-%d")
        exhibitor_profiles.append({
            "_id": f"XP{xp_id}",
            "exhibitor": exhibitor["_id"],
            "event": ev["_id"],
            "eventName": ev["name"],
            "eventDate": event_date,
            "boothInfo": {"size": random.choice(["S", "M", "L"])},
            "hostessRequirements": random.randint(1, 8),
            "contactInfo": {"email": exhibitor["contactEmail"], "phone": exhibitor["phone"]},
            "representativePhoto": None,
            "businessCard": None
        })
        xp_id += 1
db["ExhibitorProfile"].insert_many(exhibitor_profiles)

# ---------- EventHostUser Collection ----------
hosts = []
for i in range(1, 101):  # 100 hosts
    hosts.append({
        "_id": f"H{i}",
        "fullName": f"Host {i}",
        "gender": random.choice(["M", "F"]),
        "age": random.randint(20, 40),
        "nationality": random.choice(["UAE", "EG", "IN", "PH"]),
        "phone": rand_phone(),
        "whatsapp": rand_phone(),
        "languages": random.sample(languages, k=random.randint(1, len(languages))),
        "yearsOfExperience": random.randint(0, 6),
        "eventTypes": random.sample(event_types, k=random.randint(1, 2)),
        "industries": random.sample(industries, k=random.randint(1, 2)),
        "certifications": random.sample(certifications, k=random.randint(0, 2)),
        "interests": ["Networking", "Sales"],
        "profilePicture": None,
        "visa": random.choice(["Visit", "Resident", "Work"]),
        "residentVisaInformation": None,
        "email": f"host{i}@mail.com",
        "password": "hashed_pw",
        "status": random.choice(["pending", "active", "deactivated"])
    })
db["EventHostUser"].insert_many(hosts)

# ---------- EventHostProfile (public subset) ----------
db["EventHostProfile"].insert_many([
    {"_id": f"HP{h['_id']}", "eventHost": h["_id"], "fullName": h["fullName"], "languages": h["languages"],
     "yearsOfExperience": h["yearsOfExperience"], "industries": h["industries"]}
    for h in hosts
])

# ---------- UnavailableDate Collection ----------
unavailable = []
for h in hosts:
    # 6 random unavailable dates over the past year
    for j in range(6):
        month_offset = random.randint(0, 12)
        base = today - timedelta(days=30*month_offset)
        d = base.replace(day=random.randint(1, 28)).strftime("%Y-%m-%d")
        unavailable.append({
            "_id": f"U{h['_id']}-{j}",
            "eventHost": h["_id"],
            "dates": [d],
            "createdAt": datetime.utcnow()
        })
db["UnavailableDate"].insert_many(unavailable)

# ---------- EventInvite Collection ----------
invites = []
for i, xp in enumerate(exhibitor_profiles, start=1):
    ev_id = xp["event"]
    exhibitor_id = xp["exhibitor"]
    # Invite between 1-5 hosts
    for j in range(random.randint(1, 5)):
        host_id = random.choice(hosts)["_id"]
        invites.append({
            "_id": f"INV{i}-{j+1}",
            "event": ev_id,
            "exhibitor": exhibitor_id,
            "eventHost": host_id,
            "otp": str(random.randint(100000, 999999)),
            "status": random.choice(["In Progress", "Accepted", "Declined"]),
            "createdAt": datetime.utcnow()
        })
        # Add participant to event
        event_doc = next(e for e in events if e["_id"] == ev_id)
        event_doc["participants"].append(host_id)

# persist updated participants
for e in events:
    db["Event"].update_one({"_id": e["_id"]}, {"$set": {"participants": e["participants"]}})
db["EventInvite"].insert_many(invites)

# ---------- ActivityLog Collection ----------
logs = []
for i in range(1, 21):
    logs.append({
        "_id": f"L{i}",
        "userId": random.choice([random.choice(hosts)["_id"], random.choice(exhibitors)["_id"]]),
        "userType": random.choice(["EventHostUser", "ExhibitorUser"]),
        "action": random.choice(["login", "create_invite", "update_profile"]),
        "details": {},
        "createdAt": datetime.utcnow() - timedelta(days=random.randint(0, 60))
    })
db["ActivityLog"].insert_many(logs)

# ---------- SupportRequest Collection ----------
tickets = []
for i in range(1, 11):
    tickets.append({
        "_id": f"SR{i}",
        "fullName": f"User {i}",
        "email": f"user{i}@mail.com",
        "phone": rand_phone(),
        "topic": random.choice(["Account", "Billing", "Platform"]),
        "message": "Help needed",
        "status": random.choice(["New", "In Progress", "Completed"]),
        "refNumber": f"REF{i:04d}",
        "userType": random.choice(["EventHostUser", "ExhibitorUser"]),
        "userId": random.choice([random.choice(hosts)["_id"], random.choice(exhibitors)["_id"]])
    })
db["SupportRequest"].insert_many(tickets)

# ---------- Ratings Collection ----------
ratings = []
for i in range(1, 31):
    ev = random.choice(events)
    ex = random.choice(exhibitors)
    h = random.choice(hosts)
    ratings.append({
        "_id": f"R{i}",
        "event": ev["_id"],
        "eventHost": h["_id"],
        "exhibitor": ex["_id"],
        "rating": random.randint(1, 5),
        "comment": "Great"
    })
db["Ratings"].insert_many(ratings)

print("Mock data generated for MongoDB (new schema)!")
