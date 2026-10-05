import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "age_group","gender","location","education","employment","income",
    "smartphone","computer","internet_access","internet_quality",
    "internet_usage","digital_literacy","online_banking","online_education",
    "online_services","social_media","internet_cost","digital_skills","barrier"
]

def calculate_score(row):
    maps = {
        "smartphone":{"Yes":15,"No":0},
        "computer":{"Yes":10,"No":0},
        "internet_access":{"Always":20,"Sometimes":10,"No":0},
        "internet_quality":{"Excellent":10,"Good":8,"Average":5,"Poor":2,"No Internet":0},
        "digital_literacy":{"High":10,"Medium":6,"Low":2},
        "digital_skills":{"High":10,"Medium":6,"Low":2},
        "online_banking":{"Yes":3,"No":0},
        "online_education":{"Yes":3,"No":0},
        "online_services":{"Yes":3,"No":0},
        "social_media":{"Yes":2,"No":0},
        "internet_cost":{"Low":4,"Medium":2,"High":0}
    }
    return float(sum(maps[k].get(str(row.get(k, "")), 0) for k in maps))

def category(score):
    if score < 40:
        return "High Digital Divide / Low Digital Inclusion"
    if score < 70:
        return "Moderate Digital Divide"
    return "Low Digital Divide / High Digital Inclusion"

def preprocess_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df.drop_duplicates().reset_index(drop=True)
    for c in REQUIRED_COLUMNS:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    df["digital_divide_score"] = df.apply(calculate_score, axis=1)
    df["category"] = df["digital_divide_score"].apply(category)
    df["created_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return df
