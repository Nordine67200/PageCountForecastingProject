import streamlit as st
import requests
from datetime import datetime, time

st.set_page_config(page_title="Predict Page Count", layout="centered")

st.title("📄 PageCount forecasting")

# Config API
api_url = "http://127.0.0.1:8000/predict"


with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        title = st.text_input("TITLE *", value="Insurance mediation (recast)")
        doc_type = st.text_input("DOC_TYPE", value="PF")
        role = st.text_input("ROLE", value="MAIN")
    with col2:
        dossier_type = st.text_input("DOSSIER_TYPE", value="ECON")
        proc_type = st.text_input("PROC_TYPE", value="COD")
        doc_ep_template = st.text_input("DOC_EP_TEMPLATE", value="OTHR")

    with col3:
        created_date = st.date_input("CREATED_1 (date)", value=datetime(2021, 6, 11))
        proc_nature = st.text_input("PROC_NATURE", value="RCST")
        committee_1 = st.text_input("COMMITTEE_1", value="ECON")

    display_debug = st.checkbox("Display debugging mode", value=True)

    submitted = st.form_submit_button("Predict")

def clean_value(v: str):
    if isinstance(v, str) and v.strip() == "":
        return None
    return v

if submitted:
    if not title.strip():
        st.error("TITLE is mandatory.")
        st.stop()

    # build the datetime
    created_dt = datetime.combine(created_date, time(0, 0, 0))
    # format string
    created_str = created_dt.isoformat(timespec="seconds")

    payload = {
        "TITLE": title,
        "CREATED_1": created_str,
        "DOC_TYPE": doc_type,
        "DOSSIER_TYPE": dossier_type,
        "PROC_TYPE": proc_type,
        "PROC_NATURE": proc_nature,
        "ROLE": role,
        "DOC_EP_TEMPLATE": doc_ep_template,
        "COMMITTEE_1": committee_1,
    }

    if display_debug:
        st.subheader("Payload sent...")
        st.json(payload)

    try:
        r = requests.post(api_url, json=payload, timeout=30)
        if r.status_code >= 400:
            st.error(f"API error: ({r.status_code})")
            st.text(r.text)
        else:

            try:
                data = r.json()

                st.subheader("Result:")
                st.success(f"📄 Net SPA predicted : {data['net_spa']:.2f}")
                if display_debug:
                    st.json(r.json())
            except Exception:
                st.text(r.text)
    except requests.RequestException as e:
        st.error("Network Issue!")
        st.write(e)
