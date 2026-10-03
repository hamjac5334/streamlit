import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="LinkedIn User Predictor", layout="centered")

INCOME_LABELS = {
    1: "Less than $10k", 2: "$10k–20k", 3: "$20k–30k", 4: "$30k–40k",
    5: "$40k–50k", 6: "$50k–75k", 7: "$75k–100k", 8: "$100k–150k", 9: "$150k+",
}
EDUC_LABELS = {
    1: "Less than high school", 2: "High school incomplete", 3: "High school graduate",
    4: "Some college, no degree", 5: "Two-year associate degree",
    6: "Four-year college degree", 7: "Some postgraduate schooling",
    8: "Postgraduate or professional degree",
}
FEATURES = ["income", "education", "parent", "married", "female", "age"]


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


@st.cache_data
def load_data():
       s = pd.read_csv(Path(__file__).parent / "social_media_usage.csv")
    ss = pd.DataFrame({
        "sm_li": clean_sm(s["web1h"]),
        "income": np.where(s["income"] <= 9, s["income"], np.nan),
        "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
        "parent": np.where(s["par"] > 2, np.nan, clean_sm(s["par"])),
        "married": np.where(s["marital"] > 6, np.nan, clean_sm(s["marital"])),
        "female": np.where(s["gender"] > 3, np.nan, np.where(s["gender"] == 2, 1, 0)),
        "age": np.where(s["age"] <= 98, s["age"], np.nan),
    })
    return ss.dropna()


@st.cache_resource
def train_model(ss):
    model = LogisticRegression(class_weight="balanced", max_iter=5000)
    model.fit(ss[FEATURES], ss["sm_li"])
    return model


ss = load_data()
model = train_model(ss)

st.title("Who uses LinkedIn?")
st.write("Enter a person's profile to predict whether they use LinkedIn.")

# ---- Inputs ----
with st.sidebar:
    st.header("Person profile")
    income = st.selectbox("Household income", list(INCOME_LABELS),
                          index=7, format_func=lambda k: INCOME_LABELS[k])
    education = st.selectbox("Education", list(EDUC_LABELS),
                             index=6, format_func=lambda k: EDUC_LABELS[k])
    age = st.slider("Age", 18, 98, 42)
    female = st.radio("Gender", ["Female", "Not female"]) == "Female"
    married = st.radio("Married?", ["Yes", "No"]) == "Yes"
    parent = st.radio("Parent of child under 18?", ["No", "Yes"]) == "Yes"

person = pd.DataFrame([{
    "income": income, "education": education, "parent": int(parent),
    "married": int(married), "female": int(female), "age": age,
}])[FEATURES]

prob = model.predict_proba(person)[0, 1]
pred = model.predict(person)[0]

# ---- Prediction ----
col1, col2 = st.columns(2)
col1.metric("Classification", "LinkedIn user" if pred == 1 else "Not a LinkedIn user")
col2.metric("Probability of using LinkedIn", f"{prob:.1%}",
            delta=f"{prob - ss['sm_li'].mean():+.1%} vs. average respondent")
st.progress(float(prob))

# ---- How probability changes with age ----
st.subheader("How age changes the prediction")
ages = pd.DataFrame(np.repeat(person.values, 81, axis=0), columns=FEATURES)
ages["age"] = np.arange(18, 99)
ages["Probability"] = model.predict_proba(ages[FEATURES])[:, 1]
st.line_chart(ages.set_index("age")["Probability"])
st.caption("Same profile as above, varying only age.")

# ---- How probability changes with income ----
st.subheader("How income changes the prediction")
inc = pd.DataFrame(np.repeat(person.values, 9, axis=0), columns=FEATURES)
inc["income"] = np.arange(1, 10)
inc["Probability"] = model.predict_proba(inc[FEATURES])[:, 1]
inc["Income"] = [f"{i}. {INCOME_LABELS[i]}" for i in inc["income"]]
st.bar_chart(inc.set_index("Income")["Probability"])

# ---- Survey context for the marketing team ----
with st.expander("LinkedIn usage in the survey data"):
    st.write(f"{len(ss):,} respondents; {ss['sm_li'].mean():.1%} use LinkedIn.")
    by_educ = ss.groupby("education")["sm_li"].mean().rename(index=EDUC_LABELS)
    st.bar_chart(by_educ.rename("Share using LinkedIn"))
