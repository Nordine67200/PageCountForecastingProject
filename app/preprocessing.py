# app/preprocessing.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

from tqdm import tqdm
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from scipy.stats import pointbiserialr

from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer
from wordcloud import STOPWORDS

from .config import settings
import joblib

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

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

def simple_tokenize(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9àâäéèêëïîôöùûüç'\s-]", " ", text)
    tokens = text.split()
    return tokens

def get_ngrams(tokens, n):
    return [" ".join(tg) for tg in ngrams(tokens, n)]

def title_to_vec(tokens, model, dim):
    valid_tokens = [t for t in tokens if t in model.wv]
    if not valid_tokens:
        return np.zeros(dim)
    return np.mean([model.wv[t] for t in valid_tokens], axis=0)

def preprocess_raw_data(
    input_path: Path | str,
) -> pd.DataFrame:
    input_path = Path(input_path)

    # 1. LOAD
    df_OSS = pd.read_excel(input_path)

    processed_parquet = Path(settings.DATA_DIR) / "features.parquet"
    processed_excel = Path(settings.DATA_DIR) / "features.xlsx"
    # filters
    df_OSS = df_OSS[df_OSS['NET_SPA'] != 0]
    df_OSS = df_OSS[df_OSS['NET_SPA'] < 2000]

    df_OSS['ROLE'] = df_OSS['ROLE'].fillna('OTHR').astype(str)
    df_OSS['PROC_TYPE'] = df_OSS['PROC_TYPE'].fillna('OTHR').astype(str)
    df_OSS['PROC_NATURE'] = df_OSS['PROC_NATURE'].fillna('OTHR').astype(str)
    df_OSS['DOC_EP_TEMPLATE']= df_OSS['DOC_EP_TEMPLATE'].fillna('OTHR').astype(str)

    special_committees = ['ANIT', 'PEST', 'AIDA', 'COVI', 'INGE', 'ING2', 'BECA', 'TAX3', 'PEGA']
    df_OSS['Committee_regrouped'] = df_OSS['COMMITTEE_1'].replace(special_committees, 'TEMP')
    df_OSS['Committee_regrouped'] = df_OSS['Committee_regrouped'].replace('BUDE', 'BUDG')

    df_OSS = df_OSS[df_OSS['CREATED_1'].notna()]
    df_OSS['CREATED_1'] = pd.to_datetime(
        df_OSS['CREATED_1'],
        format='%d-%b-%y %H.%M.%S.%f',
        errors='coerce'
    )

    # 3. FEATURES TEMPORELS
    df_OSS['DateOnly'] = df_OSS['CREATED_1'].dt.date
    df_OSS['Month'] = df_OSS['CREATED_1'].dt.month
    df_OSS['Year'] = df_OSS['CREATED_1'].dt.year
    df_OSS['DayOfWeek'] = df_OSS['CREATED_1'].dt.dayofweek
    df_OSS['Quarter'] = df_OSS['CREATED_1'].dt.quarter
    df_OSS['IsWeekend'] = df_OSS['DayOfWeek'].isin([5, 6]).astype(int)
    df_OSS['MonthName'] = df_OSS['CREATED_1'].dt.month_name()
    df_OSS['DayName'] = df_OSS['CREATED_1'].dt.day_name()

    # 4. AM_GROUPING
    cond_amother = (
        (df_OSS["DOC_TYPE"] == "AM") &
        (df_OSS["PROC_TYPE"].isin(["RSP", "DEA", "RPS"]))
    )

    cond_amdraftreport = (
        (df_OSS["DOC_TYPE"] == "AM") &
        (~df_OSS["PROC_TYPE"].isin(["RSP", "DEA", "RPS"])) &
        (df_OSS["ROLE"] == "MAIN")
    )

    cond_amdraftopinion = (
        (df_OSS["DOC_TYPE"] == "AM") &
        (~df_OSS["PROC_TYPE"].isin(["RSP", "RPS"])) &
        (df_OSS["ROLE"].isin(["AVI", "AHE", "OAC"]))
    )

    df_OSS["AM_GROUPING"] = np.select(
        [cond_amother, cond_amdraftreport, cond_amdraftopinion],
        ["AMother", "AMdraftReport", "AMdraftOpinion"],
        default=df_OSS["DOC_TYPE"]
    )

    # 5. MAPPINGS PROC / DOC / NATURE
    procedure_family_mapping = {
        "COD": "Legislative", "CNS": "Legislative",
        "INI": "Legislative", "INL": "Legislative",
        "NLE": "Legislative",
        "BUD": "Budgetary", "BUI": "Budgetary",
        "IMM": "Other", "APP": "Other", "RSP": "Other",
        "REG": "Other", "DEA": "Other", "ACI": "Other",
        "RPS": "Other", "DEC": "Other"
    }
    df_OSS['Procedure_Family'] = (
        df_OSS['PROC_TYPE']
        .map(procedure_family_mapping)
        .fillna('NA')
        .astype(str)
    )

    document_type_macro = {
        'PV': 'PROC_REPORT', 'PR': 'PROC_REPORT', 'PA': 'PROC_REPORT',
        'RR': 'PROC_REPORT', 'QO': 'PROC_REPORT', 'QZ': 'PROC_REPORT',
        'DT': 'ADMIN_DISC', 'DI': 'ADMIN_DISC', 'RD': 'ADMIN_DISC',
        'RE': 'ADMIN_DISC', 'AB': 'ADMIN_DISC', 'NT': 'ADMIN_DISC',
        'AM': 'AMENDMENTS',
        'CM': 'COMM_NOTES', 'AD': 'COMM_NOTES', 'AL': 'COMM_NOTES',
        'LT': 'COMM_NOTES', 'CR': 'COMM_NOTES', 'CN': 'COMM_NOTES',
        'OJ': 'OFFICIAL', 'DV': 'OFFICIAL', 'PE': 'OFFICIAL',
        'ED': 'OFFICIAL', 'MN': 'OFFICIAL',
        'SP': 'OPINION', 'NP': 'OPINION',
    }

    proc_nature_mapping = {
        'LEG': 'Legislative', 'INIT': 'Legislative',
        'STINI': 'Legislative', 'TRINI': 'Legislative',
        'BUD': 'Budgetary', 'PREBUD': 'Budgetary',
        'DISCH': 'Budgetary',
        'APPE': 'Approval',
        'ANRE': 'Request',
        'RESQ': 'Resolution',
        'MOFU': 'Motion',
        'DECL': 'Declaration',
        'MR': 'Report',
        'CNPE': 'Consultation',
        'ENQCOM': 'Enquiry',
        'DEAEX': 'DelegatedAct',
    }

    df_OSS['PROC_NATURE_MACRO'] = (
        df_OSS['PROC_NATURE']
        .map(proc_nature_mapping)
        .fillna('Other')
    )

    df_OSS['Document_Type_Macro'] = (
        df_OSS['DOC_TYPE']
        .map(document_type_macro)
        .fillna('OTHR')
    )

    df_OSS['PROC_DOC_COMBO'] = df_OSS['PROC_TYPE'].astype(str) + "_" + df_OSS['DOC_EP_TEMPLATE'].astype(str)
    df_OSS['PROC_DOC_TYPE'] = df_OSS['PROC_TYPE'].astype(str) + "_" + df_OSS['DOC_TYPE'].astype(str)
    df_OSS['PROC_TYPE_NATURE'] = df_OSS['PROC_TYPE'].astype(str) + "_" + df_OSS['PROC_NATURE'].astype(str)
    df_OSS['DOC_DOCEP_COMBO'] = df_OSS['DOC_TYPE'].astype(str) + "_" + df_OSS['DOC_EP_TEMPLATE'].astype(str)
    df_OSS['DOC_TYPE_PROCNATURE'] = df_OSS['DOC_TYPE'].astype(str) + "_" + df_OSS['PROC_NATURE'].astype(str)

    # 6. NLP : n-grams + corr + colonnes binaires
    df_OSS["tokens"] = df_OSS["TITLE"].astype(str).map(preprocess_text)
    n = 3
    titre = f"{n}gram"
    df_OSS[titre] = df_OSS["tokens"].map(lambda x: get_ngrams(x, n))

    all_ngrams = [ng for ngs in df_OSS[titre] for ng in ngs]
    ngram_counts = pd.Series(all_ngrams).value_counts()
    common_ngrams = ngram_counts[ngram_counts >= 3].index.tolist()

    rows_data = []
    for ngs in tqdm(df_OSS[titre], desc="Build of nnary matrix...."):
        present = {ng: 1 for ng in ngs if ng in common_ngrams}
        rows_data.append(present)

    nnary_matrix = pd.DataFrame(rows_data).fillna(0).astype(np.uint8)
    nnary_matrix = nnary_matrix.astype(pd.SparseDtype("uint8", 0))

    correlations = []
    net_pages = df_OSS["NET_SPA"].values

    for ng in tqdm(nnary_matrix.columns, desc="Calcul corrélations"):
        col = nnary_matrix[ng].values
        if col.sum() > 0:
            r, p = pointbiserialr(col, net_pages)
            correlations.append({"ngram": ng, "r": r, "p": p})

    corr_df = pd.DataFrame(correlations).dropna()
    corr_df["abs_r"] = corr_df["r"].abs()
    corr_df = corr_df.sort_values("abs_r", ascending=False)

    top_corr = corr_df.head(10)
    top_ngrams = list(top_corr["ngram"])

    for ng in top_ngrams:
        df_OSS[f'TITLE_ngram_{ng}'] = df_OSS["TITLE"].str.contains(ng, case=False).astype(int)

    # 7. Word2Vec + PCA
    df_OSS['TITLE_TOKENS'] = df_OSS['TITLE'].astype(str).apply(simple_tokenize)

    sentences = df_OSS['TITLE_TOKENS'].tolist()
    w2v_model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=100
    )
    dim = w2v_model.vector_size

    title_embeddings = df_OSS['TITLE_TOKENS'].apply(lambda toks: title_to_vec(toks, w2v_model, dim))
    title_emb_matrix = np.vstack(title_embeddings.values)

    pca_dim_w = 30
    pca = PCA(n_components=pca_dim_w, random_state=42)
    title_emb_reduced = pca.fit_transform(title_emb_matrix)

    for i in range(pca_dim_w):
        df_OSS[f'TITLE_W2V_{i+1}'] = title_emb_reduced[:, i]

    # 8. SBERT + PCA
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    titles = df_OSS['TITLE'].fillna("").astype(str).tolist()

    title_embeddings = sbert_model.encode(
        titles,
        show_progress_bar=True,
        batch_size=64
    )
    title_embeddings = np.array(title_embeddings)

    pca_dim_b = 10
    pca_sbert = PCA(n_components=pca_dim_b, random_state=42)
    title_emb_reduced = pca_sbert.fit_transform(title_embeddings)

    for i in range(pca_dim_b):
        df_OSS[f'TITLE_SBERT_{i+1}'] = title_emb_reduced[:, i]

    # 9. Features simples de TITLE
    df_OSS['TITLE_FREQ'] = df_OSS['TITLE'].map(df_OSS['TITLE'].value_counts())
    df_OSS['TITLE_WORD_COUNT'] = df_OSS['TITLE'].astype(str).str.split().str.len()
    df_OSS['TITLE_CHAR_COUNT'] = df_OSS['TITLE'].astype(str).str.len()

    # 10. MERGE AM
    df_OSS = merge_amendments(df_OSS)

    # 11. Save for predict
    models_dir = Path(settings.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(w2v_model, models_dir / "w2v_title.model")
    joblib.dump(pca, models_dir / "w2v_pca.pkl")
    joblib.dump(pca_sbert, models_dir / "sbert_pca.pkl")
    joblib.dump(top_ngrams, models_dir / "title_top_ngrams.pkl")

    # Sauvegarde parquet
    df_OSS.to_parquet(processed_parquet, index=False)

    # Sauvegarde excel
    df_OSS.to_excel(processed_excel, index=False)
    return df_OSS
