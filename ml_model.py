from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "digital_divide_model.pkl"

FEATURES = ["age_group","gender","location","education","employment","income",
            "smartphone","computer","internet_access","internet_quality","internet_usage",
            "digital_literacy","online_banking","online_education","online_services",
            "social_media","internet_cost","digital_skills","barrier"]

def train_model(df):
    df = df.dropna(subset=FEATURES+["category"]).copy()
    X, y = df[FEATURES], df["category"]
    if y.nunique() < 2 or len(df) < 30:
        return {"accuracy": 0, "report": "Not enough class diversity for training.", "matrix": []}
    cat = [c for c in FEATURES]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    model = Pipeline([("preprocessor", pre), ("classifier", DecisionTreeClassifier(max_depth=6, random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    acc=accuracy_score(y_test,pred)
    report=classification_report(y_test,pred,zero_division=0)
    labels=sorted(y.unique())
    matrix=confusion_matrix(y_test,pred,labels=labels).tolist()
    MODEL_PATH.parent.mkdir(exist_ok=True); joblib.dump(model, MODEL_PATH)
    return {"accuracy": round(acc*100,2), "report": report, "matrix": matrix, "labels": labels}

def train_from_csv(path):
    return train_model(pd.read_csv(path))

def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
