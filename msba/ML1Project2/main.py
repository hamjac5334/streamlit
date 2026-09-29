"""
Taxi cancellation early-warning app
Streamlit front end for ML1 Project 2 (Jack Hamilton & Chris Kim).

Run:  streamlit run app.py
Data: put Taxi-cancellation-case.csv next to this file, or upload it in the sidebar.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Taxi Cancellation Early Warning", layout="wide")

DATA_FILE = Path(__file__).parent / "Taxi-cancellation-case.csv"
TRAVEL_TYPES = {1: "Long distance", 2: "Point to point", 3: "Hourly rental"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- data + pipeline
@st.cache_data
def load_data(file):
    d = pd.read_csv(file)
    for col in ["from_date", "to_date", "booking_created"]:
        d[col] = pd.to_datetime(d[col], format="%m/%d/%Y %H:%M")
    return d.drop(columns=["row#"], errors="ignore")


def base_features(d):
    return pd.DataFrame({
        "point_to_point": (d["travel_type_id"] == 2).astype(int),
        "hourly_rental": (d["travel_type_id"] == 3).astype(int),
        "online_booking": d["online_booking"],
        "mobile_site_booking": d["mobile_site_booking"],
        "from_lat": d["from_lat"],
        "from_long": d["from_long"],
    }, index=d.index)


def vehicle_dummies(d, common_models, columns=None):
    model = d["vehicle_model_id"].astype(str).where(d["vehicle_model_id"].isin(common_models), "other")
    dummies = pd.get_dummies("model_" + model).astype(int)
    if columns is None:
        return dummies.drop(columns=["model_12"], errors="ignore")
    return dummies.reindex(columns=columns, fill_value=0)


def add_features(d, X):
    X = X.copy()
    lead = (d["from_date"] - d["booking_created"]).dt.total_seconds() / 3600
    X["pickup_hour"] = d["from_date"].dt.hour
    d_lat = (d["to_lat"] - X["from_lat"]) * 111.0
    d_long = (d["to_long"] - X["from_long"]) * 111.0 * np.cos(np.radians(13.0))
    X["trip_km"] = np.sqrt(d_lat ** 2 + d_long ** 2).fillna(0)
    X["log_lead_hours"] = np.log1p(lead.clip(lower=0))
    return X.drop(columns=["point_to_point"])


@st.cache_resource
def build_model(bookings, k):
    """Replicates notebook Q1-Q8: split, features, KNN imputation, KNN classifier, CV risk scores."""
    train, test = train_test_split(bookings, test_size=0.2, random_state=42,
                                   stratify=bookings["Car_Cancellation"])
    counts = train["vehicle_model_id"].value_counts()
    common_models = counts[counts >= 100].index.tolist()

    veh_train = vehicle_dummies(train, common_models)
    X_train = pd.concat([base_features(train), veh_train], axis=1)
    X_test = pd.concat([base_features(test), vehicle_dummies(test, common_models, veh_train.columns)], axis=1)

    prep = make_pipeline(StandardScaler(), KNNImputer(n_neighbors=31)).fit(X_train)
    scaler = prep.named_steps["standardscaler"]
    fill = lambda X: pd.DataFrame(scaler.inverse_transform(prep.transform(X)),
                                  columns=X.columns, index=X.index).round(6)
    X_train, X_test = add_features(train, fill(X_train)), add_features(test, fill(X_test))
    y_train, y_test = train["Car_Cancellation"], test["Car_Cancellation"]

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
    folds = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_risk = cross_val_predict(knn, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    test_risk = knn.predict_proba(X_test)[:, 1]

    return dict(train=train, test=test, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                knn=knn, common_models=common_models, cv_neighbors=(cv_risk * k).round().astype(int),
                test_neighbors=(test_risk * k).round().astype(int))


def metrics(y, flag):
    return {"Flagged": f"{flag.sum()} ({100 * flag.mean():.1f}%)",
            "Accuracy": f"{accuracy_score(y, flag):.3f}",
            "Precision": f"{precision_score(y, flag, zero_division=0):.3f}",
            "Recall": f"{recall_score(y, flag, zero_division=0):.3f}"}


def rate_chart(series, x_title, baseline, sort=None):
    df = series.rename("rate").reset_index()
    df.columns = ["group", "rate"]
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("group:N", title=x_title, sort=sort), y=alt.Y("rate:Q", title="Share cancelled", axis=alt.Axis(format="%")),
        tooltip=["group", alt.Tooltip("rate:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"y": [baseline]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    return (bars + rule).properties(height=260)

st.title("Taxi Cancellation Risk Projection")


bookings = load_data(DATA_FILE)

k = 21                     
elevated_min, high_min = 3, 6  

with st.spinner("Training model..."):
    m = build_model(bookings, k)
y_train, y_test = m["y_train"], m["y_test"]


def tier_of(n):
    return np.where(n >= high_min, "High", np.where(n >= elevated_min, "Elevated", "Standard"))


tab_score = st.container()

with tab_score:
    st.header("Score a new booking")
    Xtr = m["X_train"]
    with st.form("booking"):
        a, b, c = st.columns(3)
        travel = a.selectbox("Trip type", [2, 3, 1], format_func=TRAVEL_TYPES.get)
        channel = a.radio("Booked via", ["Phone / other", "Website", "Mobile site"])
        vehicle = a.selectbox("Vehicle model", m["common_models"] + ["other"])
        lead_hours = b.number_input("Hours between booking and pickup", 0.0, 2000.0, 5.0, step=0.5)
        pickup_hour = b.slider("Pickup hour", 0, 23, 18)
        trip_km = b.number_input("Straight-line trip distance (km)", 0.0, 60.0, 8.0,
                                 help="Only used for point-to-point trips; others are 0 km.")
        from_lat = c.number_input("Pickup latitude", 12.7, 13.4, float(Xtr["from_lat"].median()), format="%.4f")
        from_long = c.number_input("Pickup longitude", 77.3, 77.9, float(Xtr["from_long"].median()), format="%.4f")
        submitted = st.form_submit_button("Score booking", type="primary")

    if submitted:
        row = {col: 0.0 for col in Xtr.columns}
        row.update(hourly_rental=int(travel == 3), online_booking=int(channel == "Website"),
                   mobile_site_booking=int(channel == "Mobile site"), from_lat=from_lat, from_long=from_long,
                   pickup_hour=pickup_hour, trip_km=trip_km if travel == 2 else 0.0,
                   log_lead_hours=np.log1p(lead_hours))
        if f"model_{vehicle}" in row:
            row[f"model_{vehicle}"] = 1
        x = pd.DataFrame([row])[Xtr.columns]
        n = int(round(m["knn"].predict_proba(x)[0, 1] * k))
        tier = tier_of(np.array([n]))[0]
        color = {"High": "red", "Elevated": "orange", "Standard": "green"}[tier]
        r1, r2, r3 = st.columns(3)
        r1.metric("Neighbors that cancelled", f"{n} of {k}")
        r2.metric("Risk score", f"{n / k:.0%}", f"{n / k - y_train.mean():+.0%} vs average", delta_color="inverse")
        r3.markdown(f"### Tier: :{color}[{tier}]")

        scaler, clf = m["knn"].named_steps["standardscaler"], m["knn"].named_steps["kneighborsclassifier"]
        _, idx = clf.kneighbors(scaler.transform(x))
        nb = m["train"].iloc[idx[0]]
        show = pd.DataFrame({
            "Trip type": nb["travel_type_id"].map(TRAVEL_TYPES).values,
            "Vehicle": nb["vehicle_model_id"].values,
            "Lead hours": np.expm1(Xtr.iloc[idx[0]]["log_lead_hours"]).round(1).values,
            "Pickup hour": Xtr.iloc[idx[0]]["pickup_hour"].astype(int).values,
            "Trip km": Xtr.iloc[idx[0]]["trip_km"].round(1).values,
            "Cancelled": nb["Car_Cancellation"].map({1: "Yes", 0: "No"}).values,
        })
        with st.expander(f"The {k} most similar past bookings"):
            st.dataframe(show, width="stretch", hide_index=True)
