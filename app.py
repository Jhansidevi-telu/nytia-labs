import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Nytia Labs — Health Risk Demo", page_icon="🏥", layout="centered")

@st.cache_resource
def load_bundle():
    with open("nytia_demo.pkl", "rb") as f:
        return pickle.load(f)

b = load_bundle()
model           = b["model"]
le              = b["label_encoder"]
FEATURE_ORDER   = b["feature_order"]
ALL_DOMAINS     = b["all_domains"]
combo_lookup    = b["combo_lookup"]
single_profiles = b["single_domain_profiles"]

# Sentiment -> intervention urgency (matches notebook Phase 13.1)
def get_intervention_urgency(c):
    if c > -0.20: return "Monitor"
    if c > -0.40: return "Preventive"
    if c > -0.60: return "Urgent"
    return "Critical"

LABELS = {
    "nutrition": "Nutrition", "obesity": "Obesity", "sleep": "Sleep",
    "depression": "Depression", "wellness": "Wellness",
    "anti_stress": "Stress", "anti_smoke": "Smoking", "movement": "Movement",
}

st.title("🏥 Health Risk & Disease Prediction")
st.caption("Nytia Labs capstone demo — select a patient's declining health domains.")

selected = [d for d in ALL_DOMAINS if st.checkbox(LABELS[d], key=d)]

if st.button("Predict", type="primary"):
    if not selected:
        st.warning("Select at least one domain.")
        st.stop()

    key = tuple(sorted(selected))
    if key in combo_lookup:
        feats = dict(combo_lookup[key])          # exact match in training data
    else:
        # fallback: average each selected domain's profile
        rows = [single_profiles[d] for d in selected if d in single_profiles]
        feats = {f: float(np.mean([r[f] for r in rows])) for f in FEATURE_ORDER}
    feats["domain_count"] = len(selected)        # reflect actual selection

    X = np.array([[feats[f] for f in FEATURE_ORDER]])
    disease = le.inverse_transform(model.predict(X))[0]
    sentiment = feats["rec_compound"]
    urgency = get_intervention_urgency(sentiment)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Condition", disease)
    c2.metric("Intervention", urgency)
    c3.metric("Sentiment", f"{sentiment:.3f}")
