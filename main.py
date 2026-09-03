from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Student Wellbeing | Social Media",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


DATA_PATH = Path(__file__).parent / \
    "Student Social Media And Mental Health Impact.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["Stress_Level"] = pd.Categorical(
        data["Stress_Level"],
        categories=["Low", "Medium", "High", "Very High"],
        ordered=True,
    )
    return data


@st.cache_resource
def build_model(data: pd.DataFrame) -> Pipeline:
    features = [
        "Age",
        "Gender",
        "Country",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Avg_Daily_Usage_Hours",
        "Study_Hours",
        "Physical_Activity_Hours",
        "Sleep_Hours_Per_Night",
        "Stress_Level",
    ]
    categorical = [
        "Gender",
        "Country",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Stress_Level",
    ]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(
                handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(data[features], data["Mental_Health_Score"])
    return model


def apply_filters_in_main(data: pd.DataFrame) -> pd.DataFrame:
    with st.expander("🔍 Filter & refine dataset", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            levels = st.multiselect(
                "Academic level",
                options=list(data["Academic_Level"].unique()),
                default=list(data["Academic_Level"].unique()),
            )
        with col2:
            platforms = st.multiselect(
                "Platforms",
                options=sorted(data["Most_Used_Platform"].unique()),
                default=sorted(data["Most_Used_Platform"].unique()),
            )
        with col3:
            stress = st.multiselect(
                "Stress levels",
                options=["Low", "Medium", "High", "Very High"],
                default=["Low", "Medium", "High", "Very High"],
            )
        with col4:
            hours = st.slider(
                "Daily social media hours",
                min_value=0.0,
                max_value=float(data["Avg_Daily_Usage_Hours"].max()),
                value=(0.0, float(data["Avg_Daily_Usage_Hours"].max())),
                step=0.1,
            )

    return data[
        data["Academic_Level"].isin(levels)
        & data["Most_Used_Platform"].isin(platforms)
        & data["Stress_Level"].isin(stress)
        & data["Avg_Daily_Usage_Hours"].between(hours[0], hours[1])
    ]


def metric_delta(value: float, baseline: float) -> str:
    difference = value - baseline
    return f"{difference:+.1f} vs average"


data = load_data()

# Custom CSS with explicitly styled dark metric titles
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-main: #fafafa;
        --card-bg: #ffffff;
        --border-color: #e4e4e7;
        --border-hover: #d4d4d8;
        --text-dark: #09090b;
        --text-muted: #71717a;
        --accent-dark: #18181b;
        --accent-hover: #27272a;
        --pill-bg: #f4f4f5;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-dark);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit sidebar elements */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(250, 250, 250, 0.8);
        backdrop-filter: blur(12px);
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: var(--text-dark);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    p, label, span {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1 {
        font-size: 2.3rem;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }

    .eyebrow {
        display: inline-block;
        background-color: var(--pill-bg);
        color: var(--text-dark);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        border: 1px solid var(--border-color);
    }

    .subtitle {
        color: var(--text-muted);
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 680px;
        margin-bottom: 1.8rem;
    }

    .note {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-left: 3px solid var(--accent-dark);
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        color: var(--text-muted);
        font-size: 0.875rem;
        margin-top: 1.2rem;
    }

    /* Modern Minimalist Cards for Metrics */
    [data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
        transition: border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--border-hover);
    }

    /* Dark title styling for metric labels (Students shown, Average wellbeing, etc.) */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] {
        color: var(--text-dark) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
    }

    /* Expander / Filters styling */
    [data-testid="stExpander"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: var(--accent-dark);
        border-color: var(--accent-dark);
        color: #ffffff !important;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: var(--accent-hover);
        border-color: var(--accent-hover);
        color: #ffffff !important;
    }

    [data-testid="stFormSubmitButton"] button {
        color: #ffffff !important;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:focus {
        color: #ffffff !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.95rem;
        padding-bottom: 0.75rem;
    }

    .stTabs [aria-selected="true"] {
        color: var(--text-dark) !important;
        border-bottom-color: var(--accent-dark) !important;
    }

    /* Progress bar custom color */
    .stProgress > div > div > div > div {
        background-color: var(--accent-dark);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="eyebrow">Student Wellbeing Insights</div>',
            unsafe_allow_html=True)
st.title("Assess mental health under social media's effect")
st.markdown(
    '<div class="subtitle">Answer a few simple questions to receive an estimate based on patterns in the student dataset.</div>',
    unsafe_allow_html=True,
)

# Filter section embedded in main content
filtered = apply_filters_in_main(data)

if filtered.empty:
    st.info("No students match these filters. Widen the selection to continue.")
    st.stop()

# Summary Metrics
average_score = data["Mental_Health_Score"].mean()
average_usage = data["Avg_Daily_Usage_Hours"].mean()
metric_columns = st.columns(4)
metric_columns[0].metric(
    "Students shown", f"{len(filtered):,}", f"{len(filtered) / len(data):.0%} of sample")
metric_columns[1].metric("Average wellbeing", f"{filtered['Mental_Health_Score'].mean():.1f} / 10",
                         metric_delta(filtered["Mental_Health_Score"].mean(), average_score))
metric_columns[2].metric("Average daily use", f"{filtered['Avg_Daily_Usage_Hours'].mean():.1f} hrs", metric_delta(
    filtered["Avg_Daily_Usage_Hours"].mean(), average_usage))
metric_columns[3].metric("Average sleep", f"{filtered['Sleep_Hours_Per_Night'].mean():.1f} hrs", metric_delta(
    filtered["Sleep_Hours_Per_Night"].mean(), data["Sleep_Hours_Per_Night"].mean()))

st.write("")

estimate_tab, overview_tab, explore_tab = st.tabs(
    ["Assessment", "Dataset overview", "Explore patterns"])

with overview_tab:
    st.subheader("What the student sample shows")
    left, right = st.columns([1.15, 0.85])
    with left:
        score_by_stress = filtered.groupby("Stress_Level", observed=False)[
            "Mental_Health_Score"].mean().round(2)
        st.caption("Average mental health score for each stress level")
        st.bar_chart(score_by_stress, color="#0f172a", height=290)
    with right:
        platform_counts = filtered["Most_Used_Platform"].value_counts().head(8)
        st.caption("Platforms used most often")
        st.bar_chart(platform_counts, color="#64748b", height=290)
    st.markdown(
        '<div class="note">The score is a descriptive measure in this dataset, not a diagnosis. Patterns show association, not causation.</div>',
        unsafe_allow_html=True,
    )

with explore_tab:
    st.subheader("Daily habits and wellbeing")
    chart_data = filtered[["Avg_Daily_Usage_Hours", "Study_Hours",
                           "Sleep_Hours_Per_Night", "Mental_Health_Score"]].copy()
    chart_data.columns = ["Social media", "Study", "Sleep", "Mental health"]
    st.caption(
        "Average daily hours and mental health score in the filtered sample")
    st.line_chart(chart_data, height=320)
    summary = (
        filtered.groupby("Purpose_Of_Use")
        .agg(
            Students=("Mental_Health_Score", "size"),
            Wellbeing=("Mental_Health_Score", "mean"),
            Usage=("Avg_Daily_Usage_Hours", "mean"),
        )
        .round(2)
        .sort_values("Wellbeing", ascending=False)
    )
    st.caption("Results grouped by main purpose for using social media")
    st.dataframe(summary, use_container_width=True)

with estimate_tab:
    st.subheader("Tell us about your usual day")
    st.caption(
        "Choose the answer that best matches you. There are no right or wrong answers.")
    with st.form("profile_form"):
        st.markdown("#### About you")
        first, second = st.columns(2)
        with first:
            st.caption(
                "Your age helps compare your answers with similar students.")
            age = st.slider("Age", min_value=15, max_value=35, value=21)
            st.caption("This is used to compare patterns across the sample.")
            gender = st.selectbox("Gender", sorted(data["Gender"].unique()))
            st.caption("Select your current stage of education.")
            academic_level = st.selectbox(
                "Academic level", sorted(data["Academic_Level"].unique()))
        with second:
            st.caption("Select the country where you currently live or study.")
            country = st.selectbox("Country", sorted(data["Country"].unique()))
            st.caption("Choose the social platform you use most often.")
            platform = st.selectbox(
                "Platform used most", sorted(data["Most_Used_Platform"].unique()))
            st.caption("Choose what you mainly use social media for.")
            purpose = st.selectbox("Main purpose", sorted(
                data["Purpose_Of_Use"].unique()))

        st.markdown("#### Your daily habits")
        habits_left, habits_right = st.columns(2)
        with habits_left:
            st.caption("Your average daily time on social media.")
            usage = st.slider("Social media use per day", 0.0, 12.0, 4.0,
                              0.1, help="Average time spent on social media each day.")
            st.caption(
                "The number of hours you usually spend studying each day.")
            study = st.slider("Study time per day", 0.0, 12.0, 4.0, 0.1)
            st.caption("The number of hours you usually sleep each night.")
            sleep = st.slider("Sleep per night", 0.0, 12.0, 7.0, 0.1)
        with habits_right:
            st.caption(
                "The number of hours you usually spend being physically active.")
            activity = st.slider(
                "Physical activity per day", 0.0, 6.0, 2.0, 0.1)
            st.caption(
                "Select the stress level that best describes your usual week.")
            stress = st.select_slider(
                "Current stress level", options=["Low", "Medium", "High", "Very High"], value="Medium")

        submitted = st.form_submit_button("Estimate wellbeing", type="primary")

    if submitted:
        profile = pd.DataFrame(
            [{
                "Age": age,
                "Gender": gender,
                "Country": country,
                "Academic_Level": academic_level,
                "Most_Used_Platform": platform,
                "Purpose_Of_Use": purpose,
                "Avg_Daily_Usage_Hours": usage,
                "Study_Hours": study,
                "Physical_Activity_Hours": activity,
                "Sleep_Hours_Per_Night": sleep,
                "Stress_Level": str(stress),
            }]
        )
        estimate = float(build_model(data).predict(profile)[0])
        st.metric("Estimated mental health score", f"{estimate:.1f} / 10")
        st.progress(max(0.0, min(1.0, estimate / 10)))
        st.markdown(
            '<div class="note">This estimate reflects patterns in the dataset and is not a diagnosis or a substitute for professional support.</div>',
            unsafe_allow_html=True,
        )
