"""
Taxi cancellation early-warning app
Streamlit front end for ML1 Project 2 (Jack Hamilton & Chris Kim).

Run:  streamlit run app.py
Data: Taxi-cancellation-case.csv in the same folder as this file.
"""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium

st.set_page_config(page_title="Taxi Cancellation Early Warning", layout="wide")

DATA_FILE = Path(__file__).parent / "Taxi-cancellation-case.csv"
TRAVEL_TYPES = {1: "Long distance", 2: "Point to point", 3: "Hourly rental"}
LAT_RANGE, LON_RANGE = (12.7, 13.4), (77.3, 77.9)   # area the model was trained on (Bangalore)


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
    """Replicates the notebook: split, features, KNN imputation, KNN classifier."""
    train, test = train_test_split(bookings, test_size=0.2, random_state=42,
                                   stratify=bookings["Car_Cancellation"])
    counts = train["vehicle_model_id"].value_counts()
    common_models = counts[counts >= 100].index.tolist()

    veh_train = vehicle_dummies(train, common_models)
    X_train = pd.concat([base_features(train), veh_train], axis=1)

    prep = make_pipeline(StandardScaler(), KNNImputer(n_neighbors=31)).fit(X_train)
    scaler = prep.named_steps["standardscaler"]
    filled = pd.DataFrame(scaler.inverse_transform(prep.transform(X_train)),
                          columns=X_train.columns, index=X_train.index).round(6)
    X_train = add_features(train, filled)
    y_train = train["Car_Cancellation"]

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
    return dict(train=train, X_train=X_train, y_train=y_train, knn=knn, common_models=common_models)


# ---------------------------------------------------------------- settings
st.title("Taxi Cancellation Risk Projection")

bookings = load_data(DATA_FILE)
k = 21                          # neighbors (same as the notebook)
elevated_min, high_min = 3, 6   # tiers: 0-2 standard, 3-5 elevated, 6+ high

with st.spinner("Training model..."):
    m = build_model(bookings, k)
y_train = m["y_train"]
Xtr = m["X_train"]

st.caption("Early-warning model for ride cancellations. Flag a booking soon after it's made so a "
           "dispatcher can confirm it, line up a backup driver, or add it to the check-in call list. "
           "KNN (k = 21) trained on 8,000 Bangalore bookings from 2013.")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Base cancellation rate", f"{y_train.mean():.1%}")
s2.metric("High tier cancels", "32%", help="6+ of 21 neighbors cancelled (test set)")
s3.metric("Elevated tier cancels", "15%", help="3-5 of 21 neighbors cancelled (test set)")
s4.metric("Top two tiers catch", "62%", help="of all cancellations, from 23% of bookings")
st.divider()


def tier_of(n):
    return np.where(n >= high_min, "High", np.where(n >= elevated_min, "Elevated", "Standard"))


# ---------------------------------------------------------------- score a booking
st.header("Score a new booking")

# pickup map (outside the form so clicks register right away)
if "lat" not in st.session_state:
    st.session_state.lat = round(float(Xtr["from_lat"].median()), 4)
    st.session_state.lon = round(float(Xtr["from_long"].median()), 4)
    st.session_state.last_click = None
    st.session_state.snapped = False

st.markdown("**Pickup location**: click the map to set it, or type coordinates in the form below.")
fmap = folium.Map(location=[12.97, 77.59], zoom_start=11)
folium.Rectangle([[LAT_RANGE[0], LON_RANGE[0]], [LAT_RANGE[1], LON_RANGE[1]]],
                 color="#999", weight=1, fill=False, tooltip="Model's training area").add_to(fmap)
folium.Marker([st.session_state.lat, st.session_state.lon], tooltip="Pickup",
              icon=folium.Icon(color="orange")).add_to(fmap)
out = st_folium(fmap, height=380, use_container_width=True,
                returned_objects=["last_clicked"], key="pickup_map")

click = out.get("last_clicked") if out else None
if click and click != st.session_state.last_click:
    st.session_state.last_click = click
    lat = min(max(click["lat"], LAT_RANGE[0]), LAT_RANGE[1])
    lon = min(max(click["lng"], LON_RANGE[0]), LON_RANGE[1])
    st.session_state.snapped = (lat, lon) != (click["lat"], click["lng"])
    st.session_state.lat, st.session_state.lon = round(lat, 4), round(lon, 4)
    st.rerun()   # moves the marker and fills in the boxes
if st.session_state.snapped:
    st.warning("That point was outside the model's training area, so it was moved to the nearest edge.")
st.caption(f"Selected: {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

with st.form("booking"):
    a, b, c = st.columns(3)
    travel = a.selectbox("Trip type", [2, 3, 1], format_func=TRAVEL_TYPES.get)
    channel = a.radio("Booked via", ["Phone / other", "Website", "Mobile site"])
    vehicle = a.selectbox("Vehicle model", m["common_models"] + ["other"])
    lead_hours = b.number_input("Hours between booking and pickup", 0.0, 2000.0, 5.0, step=0.5)
    pickup_hour = b.slider("Pickup hour", 0, 23, 18)
    trip_km = b.number_input("Straight-line trip distance (km)", 0.0, 60.0, 8.0,
                             help="Only used for point-to-point trips; others are 0 km.")
    from_lat = c.number_input("Pickup latitude", *LAT_RANGE, key="lat", format="%.4f")
    from_long = c.number_input("Pickup longitude", *LON_RANGE, key="lon", format="%.4f")
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
