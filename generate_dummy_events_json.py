import json
import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any


def _rand_hex(n: int = 24) -> str:
    return "".join(random.choices("0123456789abcdef", k=n))


def _iso_z(dt: datetime) -> str:
    # Always emit Zulu with milliseconds like 2025-08-11T00:00:00.000Z
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _pick(lst: List[str]) -> str:
    return random.choice(lst)


def generate_events(months: int = 24, seed: int = 42) -> List[dict]:
    random.seed(seed)

    locations = [
        "Dubai Exhibition Center",
        "Abu Dhabi Expo Hall",
        "Sharjah Trade Center",
        "Riyadh Convention Center",
        "Doha World Forum",
    ]
    tech_terms = [
        "Tech", "Food", "Fashion", "Auto", "AI", "FinTech", "AdTech", "MarTech",
        "Healthcare", "SaaS", "Mobile", "Cloud", "Cyber", "Energy",
    ]
    statuses = ["Scheduled", "Ended", "Ongoing"]

    # End at last month to avoid future-only series; generate backwards then sort
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)

    events: List[dict] = []
    cursor = last_month
    for i in range(months):
        year = cursor.year
        month = cursor.month
        # number of events this month
        num_events = random.randint(2, 6)
        for _ in range(num_events):
            name = f"Dubai {_pick(tech_terms)} Expo {year}"
            location = _pick(locations)
            _id = {"$oid": _rand_hex(24)}
            # Dates: day 11-20 window for variety
            start_day = random.randint(10, 20)
            start_dt = cursor.replace(day=min(start_day, 28))
            end_dt = start_dt + timedelta(days=1)
            updated_at = end_dt + timedelta(days=random.randint(1, 14), hours=random.randint(0, 23))
            participants = [
                {"$oid": _rand_hex(24)} for __ in range(random.randint(8, 16))
            ]
            desc = (
                "This event brings together SaaS, AdTech, MarTech, Internet, and mobile technology "
                "innovators for live showcases and interactive exhibits. The event features a bustling "
                "expo floor and a Technology Training Theatre, offering networking, live demonstrations, "
                "and marketing education sessions aimed at developers, brands, and tech providers."
            )

            ev = {
                "_id": _id,
                "name": name,
                "dates": [
                    {"$date": _iso_z(start_dt)},
                    {"$date": _iso_z(end_dt)},
                ],
                "location": location,
                "description": desc,
                "status": _pick(statuses),
                "participants": participants,
                "updatedAt": {"$date": _iso_z(updated_at)},
                "__v": random.randint(1, 5),
            }
            events.append(ev)

        # step back one month
        prev = (cursor - timedelta(days=1)).replace(day=1)
        cursor = prev

    # chronological order
    events.sort(key=lambda e: e["dates"][0]["$date"]) 
    return events


def generate_hosts(num_hosts: int = 5000, qualified_ratio: float = 0.6) -> List[Dict[str, Any]]:
    first_names = [
        "Lina", "Yara", "Maya", "Omar", "Ali", "Sara", "Farah", "Noura", "Huda", "Rami",
        "Leila", "Zain", "Faris", "Hassan", "Joudi", "Khaled", "Nadine", "Rasha", "Dina", "Tala",
    ]
    last_names = [
        "Aldach", "Haddad", "Saleh", "Rahman", "Khan", "Hussein", "Zayed", "Aziz", "Mustafa", "Bakr",
    ]
    nationalities = ["UAE", "Syrian", "Egyptian", "Jordanian", "Lebanese", "Indian", "Pakistani", "Filipino"]
    languages_pool = ["English", "Arabic", "French", "Hindi", "Urdu"]

    hosts: List[Dict[str, Any]] = []
    for _ in range(num_hosts):
        full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        host_id = {"$oid": _rand_hex(24)}
        years = random.randint(0, 7)
        langs = random.sample(languages_pool, k=random.randint(1, 3))
        host = {
            "_id": host_id,
            "fullName": full_name,
            "age": random.randint(19, 35),
            "nationality": random.choice(nationalities),
            "languages": langs,
            "phone": "05" + "".join(random.choices(string.digits, k=8)),
            "whatsapp": "05" + "".join(random.choices(string.digits, k=8)),
            "email": f"{full_name.lower().replace(' ', '.')}@example.com",
            "yearsOfExperience": years,
            "eventTypes": ["Expo", "Conference", "Trade Show"],
            "industries": ["Hospitality"],
            "certifications": [],
            "qualified": (random.random() < qualified_ratio),
            "__v": 0,
        }
        hosts.append(host)
    return hosts


def generate_invites(events: List[Dict[str, Any]], hosts: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed + 1)
    invites: List[Dict[str, Any]] = []
    host_ids = [h["_id"]["$oid"] for h in hosts]
    q_host_ids = [h["_id"]["$oid"] for h in hosts if h.get("qualified")]
    nq_host_ids = [h["_id"]["$oid"] for h in hosts if not h.get("qualified")]
    statuses = ["Accepted", "Rejected", "Pending"]
    for ev in events:
        event_id = ev["_id"]["$oid"]
        # assume required hosts per event 3-8; create that many invites
        req = random.randint(3, 12)
        # Bias selection: 80% qualified, 20% from the rest
        num_q = max(1, int(round(req * 0.8))) if q_host_ids else 0
        num_nq = max(0, req - num_q)
        chosen: List[str] = []
        if num_q > 0 and len(q_host_ids) >= num_q:
            chosen.extend(random.sample(q_host_ids, k=num_q))
        if num_nq > 0:
            pool = nq_host_ids if len(nq_host_ids) >= num_nq else host_ids
            chosen.extend(random.sample(pool, k=num_nq))
        invited_hosts = chosen if chosen else random.sample(host_ids, k=req)
        base_dt = datetime.strptime(ev["dates"][0]["$date"], "%Y-%m-%dT%H:%M:%S.000Z")
        created_at = base_dt - timedelta(days=random.randint(7, 30))
        for hid in invited_hosts:
            inv = {
                "_id": {"$oid": _rand_hex(24)}
                , "event": {"$oid": event_id}
                , "exhibitor": {"$oid": _rand_hex(24)}
                , "eventHost": {"$oid": hid}
                , "otp": "".join(random.choices(string.digits, k=6))
                , "status": random.choices(statuses, weights=[0.7, 0.15, 0.15])[0]
                , "createdAt": {"$date": _iso_z(created_at)}
                , "__v": 0
            }
            invites.append(inv)
    return invites


def main(out_path: str = "dummy_mongo_events_generated.json", months: int = 24, seed: int = 42, num_hosts: int = 5000):
    events = generate_events(months=months, seed=seed)
    hosts = generate_hosts(num_hosts=num_hosts, qualified_ratio=0.6)
    invites = generate_invites(events, hosts, seed=seed)
    bundle = {
        "Event": events,
        "EventHostUser": hosts,
        "EventInvite": invites,
        # empty placeholders for compatibility
        "ExhibitorUser": [],
        "ExhibitorProfile": [],
        "UnavailableDate": [],
    }
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Wrote collections to {out_path}: Event={len(events)}, EventHostUser={len(hosts)}, EventInvite={len(invites)}")


if __name__ == "__main__":
    # Simple CLI via env/args could be added; defaults chosen for convenience
    main()


