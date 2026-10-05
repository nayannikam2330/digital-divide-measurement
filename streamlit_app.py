from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from analysis import make_charts
from database import (
    delete_response,
    fetch_all,
    init_db,
    insert_response,
    replace_from_dataframe,
)
from ml_model import train_model
from preprocessing import REQUIRED_COLUMNS, calculate_score, category, preprocess_dataframe

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Digital Divide Measurement", page_icon="📊", layout="wide")
init_db()


def load_data():
    return pd.DataFrame(fetch_all())


def make_report(df):
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Digital Divide Measurement Analysis", styles["Title"]),
        Paragraph("Survey Data Analysis Report", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(f"Total respondents: {len(df)}", styles["BodyText"]),
        Paragraph(
            f"Average Digital Divide Score: {df.digital_divide_score.mean():.2f}/100",
            styles["BodyText"],
        ),
    ]
    for location in ("Rural", "Urban"):
        group = df[df.location == location]
        if not group.empty:
            access = (group.internet_access != "No").mean() * 100
            story.append(
                Paragraph(
                    f"{location}: average score {group.digital_divide_score.mean():.2f}; "
                    f"internet access {access:.1f}%",
                    styles["BodyText"],
                )
            )
    story.extend(
        [
            Spacer(1, 10),
            Paragraph("Interpretation", styles["Heading2"]),
            Paragraph(
                "Higher scores indicate stronger digital inclusion and lower digital divide. "
                "Scores below 40 indicate high digital divide, 40-69 moderate divide, "
                "and 70-100 low divide/high inclusion.",
                styles["BodyText"],
            ),
        ]
    )
    table_data = [["Category", "Respondents"]] + [
        [str(label), str(count)]
        for label, count in df.category.value_counts().items()
    ]
    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.extend([Spacer(1, 12), table])
    document.build(story)
    return output.getvalue()


st.title("Digital Divide Measurement")
st.caption("Survey-based analysis of digital access, literacy, and inclusion.")
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Survey", "Rural vs Urban", "Predictive Analysis", "Admin / Data"],
)
df = load_data()

if page == "Dashboard":
    st.header("Analysis Dashboard")
    if df.empty:
        st.info("No survey responses yet. Submit a response on the Survey page.")
    else:
        rural = df.loc[df.location == "Rural", "digital_divide_score"]
        urban = df.loc[df.location == "Urban", "digital_divide_score"]
        metrics = st.columns(4)
        metrics[0].metric("Total responses", len(df))
        metrics[1].metric("Average score", f"{df.digital_divide_score.mean():.2f}")
        metrics[2].metric("Rural score", f"{rural.mean():.2f}" if not rural.empty else "N/A")
        metrics[3].metric("Urban score", f"{urban.mean():.2f}" if not urban.empty else "N/A")

        metrics = st.columns(3)
        metrics[0].metric("Internet access", f"{(df.internet_access != 'No').mean() * 100:.1f}%")
        metrics[1].metric("Smartphone ownership", f"{(df.smartphone == 'Yes').mean() * 100:.1f}%")
        metrics[2].metric("High digital literacy", f"{(df.digital_literacy == 'High').mean() * 100:.1f}%")

        st.subheader("Visualizations")
        chart_files = make_charts(df)
        chart_items = list(chart_files.items())
        for start in range(0, len(chart_items), 2):
            columns = st.columns(2)
            for column, (label, filename) in zip(columns, chart_items[start : start + 2]):
                column.image(
                    BASE_DIR / "charts" / filename,
                    caption=label.replace("_", " ").title(),
                    width="stretch",
                )

        st.download_button(
            "Download survey data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="survey_data.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download analysis report (PDF)",
            make_report(df),
            file_name="digital_divide_report.pdf",
            mime="application/pdf",
        )

elif page == "Survey":
    st.header("Digital Divide Survey")
    st.caption("Demo survey responses should not contain personal information.")
    options = {
        "age_group": ["15-24", "25-34", "35-44", "45-54", "55+"],
        "gender": ["Male", "Female", "Other"],
        "location": ["Rural", "Urban"],
        "education": ["Primary", "Secondary", "Higher Secondary", "Graduate", "Postgraduate"],
        "employment": ["Student", "Employed", "Self-employed", "Unemployed", "Retired"],
        "income": ["Below 10000", "10000-25000", "25001-50000", "50001-100000", "Above 100000"],
        "smartphone": ["Yes", "No"],
        "computer": ["Yes", "No"],
        "internet_access": ["Always", "Sometimes", "No"],
        "internet_quality": ["Excellent", "Good", "Average", "Poor", "No Internet"],
        "internet_usage": ["Daily", "Several times/week", "Weekly", "Rarely", "Never"],
        "digital_literacy": ["High", "Medium", "Low"],
        "online_banking": ["Yes", "No"],
        "online_education": ["Yes", "No"],
        "online_services": ["Yes", "No"],
        "social_media": ["Yes", "No"],
        "internet_cost": ["Low", "Medium", "High"],
        "digital_skills": ["High", "Medium", "Low"],
        "barrier": [
            "Cost",
            "Poor Network",
            "Lack of Devices",
            "Low Digital Skills",
            "No Local Availability",
            "No Major Barrier",
        ],
    }
    with st.form("survey_form"):
        form_columns = st.columns(2)
        response = {}
        for index, field in enumerate(REQUIRED_COLUMNS):
            response[field] = form_columns[index % 2].selectbox(
                field.replace("_", " ").title(), options[field]
            )
        submitted = st.form_submit_button("Submit survey")
    if submitted:
        score = calculate_score(response)
        result_category = category(score)
        insert_response(response, score, result_category)
        st.success(f"Response saved. Inclusion score: {score:.1f}/100 — {result_category}")
        st.rerun()

elif page == "Rural vs Urban":
    st.header("Rural vs Urban Analysis")
    if df.empty:
        st.info("No survey data available yet.")
    else:
        summaries = []
        for location in ("Rural", "Urban"):
            group = df[df.location == location]
            if group.empty:
                continue
            summaries.append(
                {
                    "Location": location,
                    "Respondents": len(group),
                    "Average score": round(group.digital_divide_score.mean(), 2),
                    "Internet access %": round((group.internet_access != "No").mean() * 100, 1),
                    "Smartphone %": round((group.smartphone == "Yes").mean() * 100, 1),
                    "Computer %": round((group.computer == "Yes").mean() * 100, 1),
                    "High literacy %": round((group.digital_literacy == "High").mean() * 100, 1),
                    "High skills %": round((group.digital_skills == "High").mean() * 100, 1),
                    "Online services %": round((group.online_services == "Yes").mean() * 100, 1),
                    "Main barrier": group.barrier.value_counts().idxmax(),
                }
            )
        st.dataframe(pd.DataFrame(summaries), hide_index=True, width="stretch")
        st.info("A lower average score indicates greater digital disadvantage.")

elif page == "Predictive Analysis":
    st.header("Decision Tree Predictive Analysis")
    st.caption("Predicts the digital divide category from survey attributes.")
    if df.empty:
        st.info("No survey data is available to analyze.")
    else:
        result = train_model(df)
        if result.get("labels"):
            st.metric("Model accuracy", f"{result['accuracy']:.2f}%")
            st.text("Classification report")
            st.code(result["report"])
            st.text("Confusion matrix")
            st.dataframe(
                pd.DataFrame(
                    result["matrix"],
                    index=result["labels"],
                    columns=result["labels"],
                ),
                width="stretch",
            )
        else:
            st.warning(result["report"])

elif page == "Admin / Data":
    st.header("Admin / Data Management")
    if not st.session_state.get("is_admin"):
        with st.form("admin_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login = st.form_submit_button("Log in")
        if login:
            if username == "admin" and password == "admin123":
                st.session_state.is_admin = True
                st.rerun()
            st.error("Invalid admin credentials.")
    else:
        if st.button("Log out"):
            st.session_state.is_admin = False
            st.rerun()

        uploaded = st.file_uploader("Upload a CSV to replace the current survey records", type="csv")
        if uploaded and st.button("Validate and replace records"):
            try:
                uploaded_df = preprocess_dataframe(pd.read_csv(uploaded))
                replace_from_dataframe(uploaded_df)
                uploaded_df.to_csv(BASE_DIR / "data" / "processed_data.csv", index=False)
                st.success(f"Imported {len(uploaded_df)} survey records.")
                st.rerun()
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(f"CSV was not imported: {error}")

        df = load_data()
        st.subheader("Stored survey responses")
        if df.empty:
            st.info("There are no stored responses.")
        else:
            st.dataframe(df, hide_index=True, width="stretch")
            record_id = st.selectbox("Select response ID to delete", df.id.tolist())
            if st.button("Delete selected response"):
                delete_response(int(record_id))
                st.success(f"Deleted response {record_id}.")
                st.rerun()
