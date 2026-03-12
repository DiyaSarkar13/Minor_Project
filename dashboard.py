import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, warnings, math, re
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report, accuracy_score,
    f1_score, precision_score, recall_score,
    average_precision_score
)
warnings.filterwarnings("ignore")

st.set_page_config(page_title="NoSQLGuard", page_icon="🔗", layout="wide")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return {
        "svm"   : joblib.load("models/svm_model.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "pca"   : joblib.load("models/pca.pkl"),
        "meta"  : joblib.load("models/meta.pkl"),
    }

@st.cache_data
def load_data():
    return pd.read_csv("data/blockchain_nosql_injection.csv")

model = load_model()
df    = load_data()
y     = df["is_nosql_injection"]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🔗 NoSQLGuard")
st.sidebar.markdown("Blockchain NoSQL Injection Detection")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "📊 Dataset Overview",
    "📈 Model Performance",
    "🔗 Blockchain Intelligence",
    "🌍 Attack Map",
    "🔍 Predict a Request",
])
st.sidebar.divider()
st.sidebar.markdown("**Dataset:** 800 records · 53 features")
st.sidebar.markdown("**Normal:** 603 (75.4%)")
st.sidebar.markdown("**Attacks:** 197 (24.6%)")
st.sidebar.markdown("**CV AUC:** 0.983 ± 0.011")
st.sidebar.markdown("**Test Accuracy:** 0.944")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", "800")
    c2.metric("Features", "53")
    c3.metric("Normal", "603 (75.4%)")
    c4.metric("Attacks", "197 (24.6%)")
    c5.metric("Platforms", "5")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            values=[603, 197],
            names=["Normal", "Attack"],
            color_discrete_sequence=["#4488ff", "#ff4466"],
            hole=0.4,
            title="Class Distribution",
        )
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        plat = df[df["is_nosql_injection"] == 1]["blockchain_platform"].value_counts()
        fig = px.bar(
            x=plat.index, y=plat.values,
            color=plat.values,
            color_continuous_scale="Reds",
            title="Attacks by Blockchain Platform",
        )
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        inj = df["injection_type"].value_counts().dropna()
        fig = px.bar(
            x=inj.index, y=inj.values,
            color=inj.values,
            color_continuous_scale="Oranges",
            title="NoSQL Injection Types",
        )
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        atk = df[y == 1]["confidence"]
        nrm = df[y == 0]["confidence"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=nrm, name="Normal",
            marker_color="#4488ff", opacity=0.7, nbinsx=20))
        fig.add_trace(go.Histogram(
            x=atk, name="Attack",
            marker_color="#ff4466", opacity=0.7, nbinsx=20))
        fig.update_layout(
            barmode="overlay", title="Confidence Distribution",
            paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    filt = st.radio("Filter", ["All", "Normal", "Attacks Only"], horizontal=True)
    show = df if filt == "All" else (
           df[y == 0] if filt == "Normal" else df[y == 1])
    st.dataframe(show.head(30), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    try:
        pred_df = pd.read_csv("outputs/predictions.csv")
        y_true  = pred_df["true_label"]
        y_pred  = pred_df["predicted"]
        y_prob  = pred_df["attack_prob"]
    except:
        st.error("Run nosql_injection_ml_project.py first to generate predictions.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.4f}")
    c2.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    c3.metric("Recall",    f"{recall_score(y_true, y_pred):.4f}")
    c4.metric("F1-Score",  f"{f1_score(y_true, y_pred):.4f}")
    c5.metric("ROC-AUC",   f"{roc_auc_score(y_true, y_prob):.4f}")
    st.info("Expected: Accuracy ~0.94 | AUC ~0.97 — Realistic, not 1.0 ✅")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Normal", "Attack"],
            y=["Normal", "Attack"],
            title="Confusion Matrix",
        )
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, fill="tozeroy",
            name=f"SVM (AUC={auc:.3f})",
            line=dict(color="#00f5c4", width=2),
            fillcolor="rgba(0,245,196,0.12)"))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color="gray", dash="dash"),
            name="Random"))
        fig.update_layout(
            title="ROC Curve",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    rpt = classification_report(
        y_true, y_pred,
        target_names=["Normal", "NoSQL Injection"],
        output_dict=True)
    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — BLOCKCHAIN INTELLIGENCE
# ══════════════════════════════════════════════════════════════
elif page == "🔗 Blockchain Intelligence":
    st.title("🔗 Blockchain Intelligence")
    attacks = df[df["is_nosql_injection"] == 1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Attacks",       str(len(attacks)))
    c2.metric("Platforms Affected",  str(attacks["blockchain_platform"].nunique()))
    c3.metric("Smart Contracts Hit", str(attacks["smart_contract"].nunique()))
    c4.metric("Avg Confidence",      f"{attacks['confidence'].mean():.3f}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        ct = pd.crosstab(df["blockchain_platform"], df["is_nosql_injection"])
        ct.columns = ["Normal", "Attack"]
        fig = px.bar(
            ct, barmode="group",
            color_discrete_sequence=["#4488ff", "#ff4466"],
            title="Attack vs Normal by Platform")
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sc = attacks["smart_contract"].value_counts()
        fig = px.pie(
            values=sc.values, names=sc.index,
            title="Attacked Smart Contracts",
            color_discrete_sequence=px.colors.sequential.Reds)
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        cf = attacks["contract_function"].value_counts()
        fig = px.bar(
            x=cf.values, y=cf.index, orientation="h",
            color=cf.values, color_continuous_scale="Reds",
            title="Attacked Contract Functions")
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        oe = attacks["onchain_event_type"].value_counts()
        fig = px.pie(
            values=oe.values, names=oe.index,
            title="On-chain Event Types", hole=0.4,
            color_discrete_sequence=px.colors.sequential.Oranges)
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Attack Payloads")
    st.dataframe(
        attacks[["request_body", "injection_type",
                 "blockchain_platform", "confidence"]].head(10),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ATTACK MAP
# ══════════════════════════════════════════════════════════════
elif page == "🌍 Attack Map":
    st.title("🌍 Global Attack Map")
    attacks = df[df["is_nosql_injection"] == 1].copy()

    fig = px.scatter_geo(
        attacks,
        lat="geo_lat", lon="geo_lon",
        color="confidence", size="confidence",
        hover_data=["source_country", "source_ip",
                    "blockchain_platform", "injection_type"],
        color_continuous_scale="Reds",
        projection="natural earth",
        size_max=15,
        title="NoSQL Injection Attack Origins",
    )
    fig.update_layout(
        paper_bgcolor="#0e1117", height=450,
        geo=dict(bgcolor="#0e1117", landcolor="#1e2130",
                 showcountries=True, countrycolor="#333355"))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        ctry = attacks["source_country"].value_counts().head(10)
        fig = px.bar(
            x=ctry.values, y=ctry.index, orientation="h",
            color=ctry.values, color_continuous_scale="Reds",
            title="Top Source Countries")
        fig.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        df2 = df.copy()
        df2["timestamp_utc"] = pd.to_datetime(df2["timestamp_utc"])
        df2["hour"] = df2["timestamp_utc"].dt.hour
        hourly = df2.groupby("hour")["is_nosql_injection"].mean() * 100
        fig = go.Figure(go.Scatter(
            x=hourly.index, y=hourly.values,
            fill="tozeroy", mode="lines+markers",
            line=dict(color="#00f5c4", width=2)))
        fig.update_layout(
            title="Attack Rate by Hour",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            xaxis_title="Hour (UTC)",
            yaxis_title="Attack Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT A REQUEST
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Predict a Request":
    st.title("🔍 Predict a Request")
    st.markdown("Runs SVM model directly — no API needed.")

    col1, col2 = st.columns(2)
    with col1:
        request_body  = st.text_area(
            "Request Body (JSON)",
            value='{"username":{"$ne":null},"password":{"$ne":null}}',
            height=100)
        query_string  = st.text_input("Query String", value="q=status%3AOK")
        response_code = st.number_input("Response Code", value=200)
        request_bytes = st.number_input("Request Bytes", value=878)

    with col2:
        confidence = st.slider("WAF Confidence", 0.0, 1.0, 0.35, 0.01)
        latency_ms = st.number_input("Latency (ms)", value=60)
        blockchain = st.selectbox("Blockchain Platform", [
            "HyperledgerFabric", "Ethereum-Private",
            "Quorum", "Polygon-Edge", "Corda"])

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        qs = query_string
        rb = request_body

        features = {
            "confidence"       : confidence,
            "request_bytes"    : request_bytes,
            "latency_ms"       : latency_ms,
            "response_code"    : response_code,
            "response_bytes"   : 0,
            "asn"              : 0,
            "geo_lat"          : 0,
            "geo_lon"          : 0,
            "rate_limited"     : 0,
            "pii_exposure_risk": 0,
            "ledger_committed" : 0,
            "block_height"     : 0,
            "hour"             : 12,
            "minute"           : 0,
            "dayofweek"        : 0,
            "qs_length"        : len(qs),
            "rb_length"        : len(rb),
            "qs_is_empty"      : int(len(qs.strip()) == 0),
            "rb_is_empty"      : int(len(rb.strip()) == 0),
            "qs_has_pct"       : int("%" in qs),
            "rb_has_brace"     : int("{" in rb),
            "has_session"      : 0,
            "has_device"       : 0,
            "has_user"         : 0,
            "has_account"      : 0,
            "has_tx"           : 0,
        }

        meta    = model["meta"]
        X_input = pd.DataFrame([features]).reindex(
                  columns=meta["feature_names"], fill_value=0).astype(float)
        X_sc    = model["scaler"].transform(X_input)
        X_pca   = model["pca"].transform(X_sc)
        pred    = int(model["svm"].predict(X_pca)[0])
        proba   = float(model["svm"].predict_proba(X_pca)[0][1])

        if pred == 1:
            st.error("🚨 NoSQL INJECTION DETECTED")
        else:
            st.success("✅ NORMAL TRAFFIC")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prediction",  "ATTACK" if pred == 1 else "NORMAL")
        c2.metric("Attack Prob", f"{proba:.1%}")
        c3.metric("Action",      "BLOCK"  if pred == 1 else "ALLOW")
        c4.metric("Risk",        "HIGH"   if proba > 0.8 else
                                 "MEDIUM" if proba > 0.5 else "LOW")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": "Attack Probability (%)"},
            gauge={
                "axis"     : {"range": [0, 100]},
                "bar"      : {"color": "#ff4466" if pred == 1 else "#00f5c4"},
                "steps"    : [
                    {"range": [0,  40], "color": "#0d2b0d"},
                    {"range": [40, 70], "color": "#2b2b0d"},
                    {"range": [70,100], "color": "#2b0d0d"}],
                "threshold": {"line": {"color": "white", "width": 4},
                              "value": 70},
            }))
        fig.update_layout(
            paper_bgcolor="#0e1117",
            font_color="white",
            height=280)
        st.plotly_chart(fig, use_container_width=True)