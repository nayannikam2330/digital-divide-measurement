from pathlib import Path
import hashlib
import threading
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)
_CHART_LOCK = threading.Lock()
_CHART_FILES = {
    "rural_urban_score": "rural_urban_score.png",
    "score_distribution": "score_distribution.png",
    "internet_access_location": "internet_access_location.png",
    "smartphone": "smartphone.png",
    "computer": "computer.png",
    "literacy_education": "literacy_education.png",
    "usage_age": "usage_age.png",
    "score_income": "score_income.png",
    "barriers": "barriers.png",
    "category_location": "category_location.png",
}

def _save_chart(path):
    try:
        if not path.exists():
            plt.savefig(path, dpi=130)
    finally:
        plt.close()

def make_charts(df):
    if df.empty:
        return {}
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
        + repr(tuple((str(column), str(dtype)) for column, dtype in zip(df.columns, df.dtypes))).encode()
    ).hexdigest()[:16]
    filenames = {
        key: f"{Path(filename).stem}-{data_hash}{Path(filename).suffix}"
        for key, filename in _CHART_FILES.items()
    }
    with _CHART_LOCK:
        if all((CHART_DIR / filename).is_file() for filename in filenames.values()):
            return filenames
        return _make_charts(df, data_hash)

def _make_charts(df, data_hash):
    sns.set_theme(style="whitegrid")
    charts = {}

    plt.figure(figsize=(7,4))
    df.groupby("location")["digital_divide_score"].mean().plot(kind="bar")
    plt.title("Average Digital Divide Score: Rural vs Urban")
    plt.ylabel("Score (0-100)")
    plt.tight_layout(); path=CHART_DIR/f"rural_urban_score-{data_hash}.png"; _save_chart(path)
    charts["rural_urban_score"] = path.name

    plt.figure(figsize=(7,4))
    plt.hist(df["digital_divide_score"], bins=10)
    plt.title("Digital Divide Score Distribution"); plt.xlabel("Score"); plt.ylabel("Respondents")
    plt.tight_layout(); path=CHART_DIR/f"score_distribution-{data_hash}.png"; _save_chart(path)
    charts["score_distribution"] = path.name

    plt.figure(figsize=(7,4))
    pd.crosstab(df["location"], df["internet_access"], normalize="index").mul(100).plot(kind="bar", stacked=True)
    plt.title("Internet Access by Location"); plt.ylabel("Percentage"); plt.legend(title="Access")
    plt.tight_layout(); path=CHART_DIR/f"internet_access_location-{data_hash}.png"; _save_chart(path)
    charts["internet_access_location"] = path.name

    plt.figure(figsize=(6,4))
    df["smartphone"].value_counts().plot(kind="pie", autopct="%1.1f%%", ylabel="")
    plt.title("Smartphone Ownership"); plt.tight_layout()
    path=CHART_DIR/f"smartphone-{data_hash}.png"; _save_chart(path); charts["smartphone"]=path.name

    plt.figure(figsize=(6,4))
    df["computer"].value_counts().plot(kind="pie", autopct="%1.1f%%", ylabel="")
    plt.title("Computer/Laptop Ownership"); plt.tight_layout()
    path=CHART_DIR/f"computer-{data_hash}.png"; _save_chart(path); charts["computer"]=path.name

    edu = df.groupby("education")["digital_literacy"].apply(lambda s: (s=="High").mean()*100).sort_values()
    plt.figure(figsize=(8,4)); edu.plot(kind="barh")
    plt.title("High Digital Literacy by Education"); plt.xlabel("Percentage")
    plt.tight_layout(); path=CHART_DIR/f"literacy_education-{data_hash}.png"; _save_chart(path)
    charts["literacy_education"]=path.name

    usage = pd.crosstab(df["age_group"], df["internet_usage"]).reindex(index=sorted(df["age_group"].unique()), fill_value=0)
    usage.plot(kind="bar", figsize=(9,4))
    plt.title("Internet Usage by Age Group"); plt.ylabel("Respondents")
    plt.tight_layout(); path=CHART_DIR/f"usage_age-{data_hash}.png"; _save_chart(path)
    charts["usage_age"]=path.name

    income_order = ["Below 10000","10000-25000","25001-50000","50001-100000","Above 100000"]
    inc = df.groupby("income")["digital_divide_score"].mean().reindex(income_order)
    plt.figure(figsize=(8,4)); inc.plot(kind="bar")
    plt.title("Average Digital Divide Score by Income"); plt.ylabel("Score")
    plt.xticks(rotation=25); plt.tight_layout(); path=CHART_DIR/f"score_income-{data_hash}.png"; _save_chart(path)
    charts["score_income"]=path.name

    plt.figure(figsize=(8,4))
    df["barrier"].value_counts().sort_values().plot(kind="barh")
    plt.title("Main Barriers to Digital Access"); plt.xlabel("Respondents")
    plt.tight_layout(); path=CHART_DIR/f"barriers-{data_hash}.png"; _save_chart(path)
    charts["barriers"]=path.name

    plt.figure(figsize=(7,4))
    pd.crosstab(df["location"], df["category"], normalize="index").mul(100).plot(kind="bar")
    plt.title("Digital Divide Category by Location"); plt.ylabel("Percentage")
    plt.xticks(rotation=0); plt.tight_layout(); path=CHART_DIR/f"category_location-{data_hash}.png"; _save_chart(path)
    charts["category_location"]=path.name
    return charts
