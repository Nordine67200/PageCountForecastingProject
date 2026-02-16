import streamlit as st
import requests
import pandas as pd
from datetime import datetime, time
from pathlib import Path
import json

st.set_page_config(page_title="Predict Page Count", layout="centered")
st.title("📄 PageCount forecasting")

# Config API
def load_properties(path):
    props = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                props[key.strip()] = value.strip()
    return props

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "ui_config.properties"

props = load_properties(config_path)

api_url = props["api.base-url"].rstrip("/") + props["api.predict-path"]
print(f'api url: {api_url}')

# --- CONFIG DATA FILE ---
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_PATH = BASE_DIR / "data" / "SiriusOSS_export.xlsx"

# Champs à prendre dans l'Excel (TITLE exclu volontairement)
FIELDS_FROM_EXCEL = [
    "DOC_TYPE",
    "DOSSIER_TYPE",
    "PROC_TYPE",
    "PROC_NATURE",
    "ROLE",
    "DOC_EP_TEMPLATE",
    "COMMITTEE_1",
]

@st.cache_data(show_spinner=False)
def load_reference_values(excel_path: Path) -> dict[str, list[str]]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {excel_path.resolve()}")

    df = pd.read_excel(excel_path)
    available = set(df.columns)

    missing = [c for c in FIELDS_FROM_EXCEL if c not in available]
    if missing:
        raise KeyError(f"Colonnes manquantes dans l'Excel: {missing}")

    values: dict[str, list[str]] = {}
    for col in FIELDS_FROM_EXCEL:
        series = (
            df[col]
            .dropna()
            .astype(str)
            .map(lambda x: x.strip())
        )
        uniques = sorted([v for v in series.unique() if v != ""])
        values[col] = [""] + uniques

    return values

def clean_value(v: str):
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

# --- LOAD VALUES FROM EXCEL ---
try:
    ref_values = load_reference_values(EXCEL_PATH)
except Exception as e:
    st.error("Impossible de charger les valeurs depuis SiriusOSS_export.xlsx")
    st.exception(e)
    st.stop()

# --- UI FORM ---
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        title = st.text_input("TITLE *", value="Insurance mediation (recast)")
        doc_type = st.selectbox("DOC_TYPE", options=ref_values["DOC_TYPE"], index=0)
        role = st.selectbox("ROLE", options=ref_values["ROLE"], index=0)

    with col2:
        dossier_type = st.selectbox("DOSSIER_TYPE", options=ref_values["DOSSIER_TYPE"], index=0)
        proc_type = st.selectbox("PROC_TYPE", options=ref_values["PROC_TYPE"], index=0)
        doc_ep_template = st.selectbox("DOC_EP_TEMPLATE", options=ref_values["DOC_EP_TEMPLATE"], index=0)

    with col3:
        created_date = st.date_input("CREATED_1 (date)", value=datetime(2021, 6, 11))
        proc_nature = st.selectbox("PROC_NATURE", options=ref_values["PROC_NATURE"], index=0)
        committee_1 = st.selectbox("COMMITTEE_1", options=ref_values["COMMITTEE_1"], index=0)

    display_debug = st.query_params.get("debug", "false").lower() in ("1", "true", "yes", "y")

    print(f'display debug: {display_debug}')
    submitted = st.form_submit_button("Predict")

if submitted:
    if not title.strip():
        st.error("TITLE is mandatory.")
        st.stop()

    created_dt = datetime.combine(created_date, time(0, 0, 0))
    created_str = created_dt.isoformat(timespec="seconds")

    payload = {
        "TITLE": title,
        "CREATED_1": created_str,
        "DOC_TYPE": clean_value(doc_type),
        "DOSSIER_TYPE": clean_value(dossier_type),
        "PROC_TYPE": clean_value(proc_type),
        "PROC_NATURE": clean_value(proc_nature),
        "ROLE": clean_value(role),
        "DOC_EP_TEMPLATE": clean_value(doc_ep_template),
        "COMMITTEE_1": clean_value(committee_1),
    }

    if display_debug:
        st.subheader("Payload sent...")
        st.code(json.dumps(payload, indent=2), language="json")

    try:
        r = requests.post(
            api_url,
            json=payload,
            params={"debug": display_debug},
            timeout=30
        )
        if r.status_code >= 400:
            st.error(f"API error: ({r.status_code})")
            st.text(r.text)
        else:
            try:
                data = r.json()
                st.subheader("Result:")
                st.success(f"📄 Net SPA predicted : {data['net_spa']:.2f}")

                if display_debug:
                    st.subheader("Response")
                    st.json(data)
            except Exception:
                st.text(r.text)

    except requests.RequestException as e:
        st.error("Network Issue!")
        st.write(e)
