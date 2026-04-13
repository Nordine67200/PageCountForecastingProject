# app/preprocessing.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from scipy.stats import pointbiserialr

from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from wordcloud import STOPWORDS

from .config import settings
from .s3_utils import upload_file_to_s3
import joblib


models_dir = Path(settings.MODELS_DIR)
data_dir = Path(settings.DATA_DIR)

pca_dim_w = 30


def load_predict_artifacts():

    top_ngrams = joblib.load(models_dir / "title_top_ngrams.pkl")
    w2v_model = joblib.load(models_dir / "w2v_title.model")
    w2v_pca = joblib.load(models_dir / "w2v_pca.pkl")
    sbert_pca = joblib.load(models_dir / "sbert_pca.pkl")
    title_counts = joblib.load(models_dir / "title_counts.pkl")
    sbert_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    return top_ngrams, w2v_model, w2v_pca, sbert_pca, title_counts, sbert_encoder


def merge_amendments(
    df,
    title_col="TITLE",
    doc_type_col="DOC_TYPE",
    am_value="AM",
    net_col="NET_SPA"
):
    df = df.copy()
    am_mask = df[doc_type_col] == am_value

    df_am = df[am_mask]
    df_other = df[~am_mask]

    agg = {}
    for c in df.columns:
        if c == net_col:
            agg[c] = "sum"
        else:
            agg[c] = "first"

    df_am_merged = (
        df_am
        .groupby(title_col, as_index=False)
        .agg(agg)
    )

    return pd.concat([df_other, df_am_merged], ignore_index=True)


stop_words = set(stopwords.words("english")) | STOPWORDS | {
    "document", "title", "version", "european", "union",
    "parliament", "council"
}

def simple_tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9àâäéèêëïîôöùûüç'\s-]", " ", text)
    tokens = text.split()
    return tokens



def title_to_vec(tokens, model, dim):
    valid_tokens = [t for t in tokens if t in model.wv]
    if not valid_tokens:
        return np.zeros(dim)
    return np.mean([model.wv[t] for t in valid_tokens], axis=0)


def preprocess_one_record(df: pd.DataFrame) -> pd.DataFrame:

    TOP_NGRAMS, W2V_MODEL, W2V_PCA, SBERT_PCA, TITLE_COUNTS, SBERT_ENCODER = load_predict_artifacts()

    for col in ["ROLE", "PROC_TYPE", "PROC_NATURE", "DOC_EP_TEMPLATE"]:
        df[col] = df[col].fillna("OTHR").astype(str)

    special_committees = ["ANIT", "PEST", "AIDA", "COVI", "INGE", "ING2", "BECA", "TAX3", "PEGA"]
    df["Committee_regrouped"] = df["COMMITTEE_1"].replace(special_committees, "TEMP")
    df["Committee_regrouped"] = df["Committee_regrouped"].replace("BUDE", "BUDG")

    df["CREATED_1"] = pd.to_datetime(df["CREATED_1"], errors="coerce")
    if df["CREATED_1"].isna().any():
        raise ValueError("CREATED_1 invalide (format date non parsable)")

    df["DateOnly"] = df["CREATED_1"].dt.date
    df["Month"] = df["CREATED_1"].dt.month
    df["Year"] = df["CREATED_1"].dt.year
    df["DayOfWeek"] = df["CREATED_1"].dt.dayofweek
    df["Quarter"] = df["CREATED_1"].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["MonthName"] = df["CREATED_1"].dt.month_name()
    df["DayName"] = df["CREATED_1"].dt.day_name()

    cond_amother = (df["DOC_TYPE"] == "AM") & (df["PROC_TYPE"].isin(["RSP", "DEA", "RPS"]))
    cond_amdraftreport = (df["DOC_TYPE"] == "AM") & (~df["PROC_TYPE"].isin(["RSP", "DEA", "RPS"])) & (df["ROLE"] == "MAIN")
    cond_amdraftopinion = (df["DOC_TYPE"] == "AM") & (~df["PROC_TYPE"].isin(["RSP", "RPS"])) & (df["ROLE"].isin(["AVI", "AHE", "OAC"]))

    df["AM_GROUPING"] = np.select(
        [cond_amother, cond_amdraftreport, cond_amdraftopinion],
        ["AMother", "AMdraftReport", "AMdraftOpinion"],
        default=df["DOC_TYPE"]
    )

    procedure_family_mapping = {
        "COD": "Legislative", "CNS": "Legislative",
        "INI": "Legislative", "INL": "Legislative",
        "NLE": "Legislative",
        "BUD": "Budgetary", "BUI": "Budgetary",
        "IMM": "Other", "APP": "Other", "RSP": "Other",
        "REG": "Other", "DEA": "Other", "ACI": "Other",
        "RPS": "Other", "DEC": "Other"
    }
    df["Procedure_Family"] = df["PROC_TYPE"].map(procedure_family_mapping).fillna("NA").astype(str)

    document_type_macro = {
        "PV": "PROC_REPORT", "PR": "PROC_REPORT", "PA": "PROC_REPORT",
        "RR": "PROC_REPORT", "QO": "PROC_REPORT", "QZ": "PROC_REPORT",
        "DT": "ADMIN_DISC", "DI": "ADMIN_DISC", "RD": "ADMIN_DISC",
        "RE": "ADMIN_DISC", "AB": "ADMIN_DISC", "NT": "ADMIN_DISC",
        "AM": "AMENDMENTS",
        "CM": "COMM_NOTES", "AD": "COMM_NOTES", "AL": "COMM_NOTES",
        "LT": "COMM_NOTES", "CR": "COMM_NOTES", "CN": "COMM_NOTES",
        "OJ": "OFFICIAL", "DV": "OFFICIAL", "PE": "OFFICIAL",
        "ED": "OFFICIAL", "MN": "OFFICIAL",
        "SP": "OPINION", "NP": "OPINION",
    }

    proc_nature_mapping = {
        "LEG": "Legislative", "INIT": "Legislative",
        "STINI": "Legislative", "TRINI": "Legislative",
        "BUD": "Budgetary", "PREBUD": "Budgetary",
        "DISCH": "Budgetary",
        "APPE": "Approval",
        "ANRE": "Request",
        "RESQ": "Resolution",
        "MOFU": "Motion",
        "DECL": "Declaration",
        "MR": "Report",
        "CNPE": "Consultation",
        "ENQCOM": "Enquiry",
        "DEAEX": "DelegatedAct",
    }

    df["PROC_NATURE_MACRO"] = df["PROC_NATURE"].map(proc_nature_mapping).fillna("Other")
    df["Document_Type_Macro"] = df["DOC_TYPE"].map(document_type_macro).fillna("OTHR")

    df["PROC_DOC_COMBO"] = df["PROC_TYPE"].astype(str) + "_" + df["DOC_EP_TEMPLATE"].astype(str)
    df["PROC_DOC_TYPE"] = df["PROC_TYPE"].astype(str) + "_" + df["DOC_TYPE"].astype(str)
    df["PROC_TYPE_NATURE"] = df["PROC_TYPE"].astype(str) + "_" + df["PROC_NATURE"].astype(str)
    df["DOC_DOCEP_COMBO"] = df["DOC_TYPE"].astype(str) + "_" + df["DOC_EP_TEMPLATE"].astype(str)
    df["DOC_TYPE_PROCNATURE"] = df["DOC_TYPE"].astype(str) + "_" + df["PROC_NATURE"].astype(str)

    for ng in TOP_NGRAMS:
        df[f"TITLE_ngram_{ng}"] = df["TITLE"].astype(str).str.contains(ng, case=False, regex=False).astype(int)

    df["TITLE_TOKENS"] = df["TITLE"].astype(str).apply(simple_tokenize)
    dim = W2V_MODEL.vector_size
    vec = df["TITLE_TOKENS"].apply(lambda toks: title_to_vec(toks, W2V_MODEL, dim))
    mat = np.vstack(vec.values)
    reduced = W2V_PCA.transform(mat)
    for i in range(reduced.shape[1]):
        df[f"TITLE_W2V_{i+1}"] = reduced[:, i]

    titles = df["TITLE"].fillna("").astype(str).tolist()
    emb = SBERT_ENCODER.encode(titles, show_progress_bar=False)
    emb = np.array(emb)
    emb_reduced = SBERT_PCA.transform(emb)

    for i in range(emb_reduced.shape[1]):
        df[f"TITLE_SBERT_{i + 1}"] = emb_reduced[:, i]

    df["TITLE_WORD_COUNT"] = df["TITLE"].astype(str).str.split().str.len()
    df["TITLE_CHAR_COUNT"] = df["TITLE"].astype(str).str.len()

    title = df["TITLE"].astype(str)
    df["TITLE_FREQ"] = title.map(lambda t: int(TITLE_COUNTS.get(t, 1)))

    return df