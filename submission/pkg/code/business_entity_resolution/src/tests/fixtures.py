import polars as pl


def records(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


REFERENCES = records([
    ("S1-1", "Svl Tubes Pvt Ltd", "Plot No: 51 First Floor, Govindan Street, Chennai, Tamil Nadu", "India"),
    ("S1-2", "Jex First Focus", "920 Stinocher St, Solon, IA", "US"),
    ("S1-3", "Choice Transit Enterprises", "2138 Talkington Drive, Flagstaff, AZ", "US"),
    ("S1-4", "Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France", "France"),
    ("S1-5", "Laxmi Developers Private Limited", "12 MG Road, Pune, Maharashtra", "India"),
    ("S1-6", "Harbor Bakery LLC", "77 Ocean Ave, Portland, ME", "US"),
])

MENTIONS = records([
    ("S2-1", "Svl  Tubes Pvt", "PLOT NO: ##51 FIRST FLOOR, GOVINDAN STREET, CHENNAI, தமிழ்நாடு", "India"),
    ("S3-1", "Svl Ttuaesb Pvt Ltd", "No 7-67 Plot No: 51 First Floor, Chennai, TN", "India"),
    ("S2-2", "Jex First", "920 STINOCHER SAINT, SOLN, IA", "US"),
    ("S3-2", "@jexfirst", "920 Stinocher St, Solon, Iowa", "US"),
    ("S3-3", "The Choice Transit-Enterprises", "2138 Talkington Dr, Flagstaf FCDP, Arizona", "US"),
    ("S2-4", "THERMAL ET FILS", "20 R PARMENTIER, DUNKERQUE, Nord", "France"),
    ("S2-5", "लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड", "12 M G ROAD, PUNE, महाराष्ट्र", "India"),
    ("S3-9", "Jex First Focus", "920 Stinocher St, Solon, IA", "India"),  # same text, other country
    ("S2-9", "Quantum Plumbing Inc", "5 Elm St, Boise, ID", "US"),  # Distractor
])

TRUTH = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2", "S3-2"], "S1-3": ["S3-3"], "S1-4": ["S2-4"], "S1-5": ["S2-5"], "S1-6": []}
