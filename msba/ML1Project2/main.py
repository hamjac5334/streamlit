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


st.sidebar.title("Cancellation early warning")
bookings = load_data(DATA_FILE)

k = st.sidebar.select_slider("K (neighbors)", options=[5, 11, 15, 21, 31, 51], value=21,
                             help="The notebook uses k = 21.")
with st.spinner("Training model and computing cross-validated risk scores..."):
    m = build_model(bookings, k)
y_train, y_test = m["y_train"], m["y_test"]

st.sidebar.markdown("**Risk tiers** (neighbors cancelled)")
elevated_min = st.sidebar.number_input("Elevated from", 1, k, value=min(3, k))
high_min = st.sidebar.number_input("High from", elevated_min + 1, k, value=max(min(6, k), elevated_min + 1))
st.sidebar.caption("Notebook defaults at k = 21: elevated 3-5, high 6+.")


def tier_of(n):
    return np.where(n >= high_min, "High", np.where(n >= elevated_min, "Elevated", "Standard"))


tab_over, tab_feat, tab_thresh, tab_tiers, tab_score = st.tabs(
    ["Overview", "Risk drivers", "Threshold explorer", "Risk tiers", "Score a booking"])

with tab_over:
    st.header("Early-warning model for ride cancellations")
    st.write("Flag a booking as at risk soon after it's made, so a dispatcher can confirm it, "
             "line up a backup driver, or put it on the check-in call list.")
    c = st.columns(4)
    c[0].metric("Bookings", f"{len(bookings):,}")
    c[1].metric("Cancellation rate", f"{bookings['Car_Cancellation'].mean():.1%}")
    c[2].metric("Majority-class baseline", f"{1 - y_test.mean():.1%}")
    c[3].metric("Predictors", m["X_train"].shape[1])

    left, right = st.columns(2)
    with left:
        st.subheader("Bookings by trip type")
        tt = bookings.groupby("travel_type_id")["Car_Cancellation"].agg(["count", "mean"])
        tt.index = tt.index.map(TRAVEL_TYPES)
        tt.columns = ["Bookings", "Cancellation rate"]
        st.dataframe(tt.style.format({"Cancellation rate": "{:.1%}"}), width="stretch")
    with right:
        st.subheader("Cancellation rate by month")
        by_month = bookings.groupby(bookings["booking_created"].dt.month)["Car_Cancellation"].mean()
        by_month.index = [MONTHS[i - 1] for i in by_month.index]
        st.altair_chart(rate_chart(by_month, "Month booked (2013)", bookings["Car_Cancellation"].mean(),
                                   sort=MONTHS), width="stretch")
    with st.expander("Missing values (training set)"):
        miss = m["train"].isna().mean().loc[lambda s: s > 0].sort_values(ascending=False)
        st.dataframe(miss.rename("Share missing").to_frame().style.format("{:.1%}"), width="stretch")
        st.caption("to_city_id / package_id / to_* are structural (depend on trip type). to_date and "
                   "from_city_id were a recording change before July 2013 and were dropped. "
                   "from_lat / from_long were KNN-imputed (k = 31) for 13 long-distance bookings.")

with tab_feat:
    st.header("What drives cancellations (training set)")
    X, base = m["X_train"], y_train.mean()
    lead = np.expm1(X["log_lead_hours"])
    by_lead = y_train.groupby(pd.cut(lead, [-1, 1, 3, 6, 12, 24, 48, 2000],
                                     labels=["<1h", "1-3h", "3-6h", "6-12h", "12-24h", "1-2d", "2d+"]),
                              observed=True).mean()
    p2p = X["trip_km"] > 0
    by_km = y_train[p2p].groupby(pd.cut(X.loc[p2p, "trip_km"], [-1, 2, 5, 10, 20, 40, 100],
                                        labels=["<2", "2-5", "5-10", "10-20", "20-40", "40+"]),
                                 observed=True).mean()
    by_hour = y_train.groupby(X["pickup_hour"]).mean()
    by_hour.index = by_hour.index.astype(str)
    a, b = st.columns(2)
    a.altair_chart(rate_chart(by_lead, "Hours between booking and pickup", base, sort=None), width="stretch")
    b.altair_chart(rate_chart(by_km, "Trip distance, km (point to point)", base, sort=None), width="stretch")
    st.altair_chart(rate_chart(by_hour, "Pickup hour", base, sort=[str(h) for h in range(24)]),
                    width="stretch")
    st.caption("Dashed line = overall training cancellation rate.")

with tab_thresh:
    st.header("Precision / recall trade-off")
    st.write(f"The model's risk score is how many of a booking's **{k} nearest neighbors** cancelled. "
             "Lower the bar to catch more cancellations at the cost of more false alarms. "
             "The table uses cross-validated scores on the training set; the test numbers below use the bar you pick.")
    rows = []
    for bar in range(1, k // 2 + 2):
        flag = (m["cv_neighbors"] >= bar).astype(int)
        rows.append({"bar": bar, "share flagged": flag.mean(),
                     "precision": precision_score(y_train, flag, zero_division=0),
                     "recall": recall_score(y_train, flag, zero_division=0)})
    tr = pd.DataFrame(rows)
    chosen = st.slider(f"Flag when at least this many of {k} neighbors cancelled", 1, int(tr["bar"].max()),
                       value=min(4, int(tr["bar"].max())))
    long = tr.melt("bar", var_name="metric", value_name="score")
    line = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("bar:O", title=f"Minimum neighbors cancelled (of {k})"),
        y=alt.Y("score:Q", title="Score on training folds"), color="metric:N",
        tooltip=["bar", "metric", alt.Tooltip("score:Q", format=".3f")])
    vline = alt.Chart(pd.DataFrame({"bar": [chosen]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(x="bar:O")
    st.altair_chart((line + vline).properties(height=320), width="stretch")

    flag_test = (m["test_neighbors"] >= chosen).astype(int)
    st.subheader("On the held-out test set")
    cols = st.columns(4)
    for col, (name, val) in zip(cols, metrics(y_test, flag_test).items()):
        col.metric(name, val)
    cm = pd.DataFrame(confusion_matrix(y_test, flag_test), index=["Actually kept", "Actually cancelled"],
                      columns=["Not flagged", "Flagged"])
    st.dataframe(cm, width="content")

with tab_tiers:
    st.header("Three risk tiers for the ops team (test set)")
    tiers = pd.DataFrame({"tier": tier_of(m["test_neighbors"]), "cancelled": y_test.values})
    tt = tiers.groupby("tier")["cancelled"].agg(["count", "sum", "mean"]).reindex(["Standard", "Elevated", "High"])
    tt.columns = ["Bookings", "Cancellations", "Cancellation rate"]
    tt["Share of bookings"] = tt["Bookings"] / len(tiers)
    tt["Share of all cancellations"] = tt["Cancellations"] / tiers["cancelled"].sum()
    st.dataframe(tt.style.format({"Cancellation rate": "{:.1%}", "Share of bookings": "{:.1%}",
                                  "Share of all cancellations": "{:.1%}"}), width="stretch")
    top = tt.loc[["Elevated", "High"]]
    st.success(f"Acting on the Elevated + High tiers means working **{top['Share of bookings'].sum():.0%}** "
               f"of bookings to reach **{top['Share of all cancellations'].sum():.0%}** of cancellations.")
    st.altair_chart(rate_chart(tt["Cancellation rate"], "Risk tier", y_test.mean(),
                               sort=["Standard", "Elevated", "High"]), width="stretch")
    st.markdown("""
**Playbook**
- **High** – confirmation call from a dispatcher; line up a backup driver.
- **Elevated** – automatic confirmation message; route non-responders to a dispatcher.
- **Standard** – leave alone (rentals, long airport runs, phone bookings, bookings a day ahead).
""")

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
