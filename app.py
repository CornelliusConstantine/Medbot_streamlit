"""
MedBot - Medical NLP Chatbot (Streamlit version)
Hybrid retrieval-based chatbot: rule-based intent + SBERT/TF-IDF semantic search.

Cara jalan:
    Lokal   : streamlit run app.py
    Cloud   : deploy lewat share.streamlit.io, hubungkan ke repo GitHub ini

Dataset:
    Taruh file CSV di folder yang sama dengan nama `medical_qa_dataset.csv`,
    berisi minimal 3 kolom: category, question, answer.
    Kalau file tidak ditemukan, app akan menampilkan file uploader di sidebar.
"""

import re
import random
import os
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# PAGE CONFIG (harus dipanggil paling awal)
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedBot - Asisten Kesehatan",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

DATA_PATH = "medical_qa_dataset.csv"

# ──────────────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 14px; }
    .medbot-disclaimer {
        font-size: 0.8rem;
        color: #888;
        border-top: 1px solid #333;
        padding-top: 0.6rem;
        margin-top: 0.6rem;
    }
    .medbot-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background: #1f6feb22;
        color: #58a6ff;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# NLTK SETUP (cached, download once per session/container)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Menyiapkan komponen NLP...")
def setup_nltk():
    import nltk
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return True


setup_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ──────────────────────────────────────────────────────────────────────────
# TRY LOADING SBERT (fallback ke TF-IDF kalau gagal / terlalu berat)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model semantic search...")
def load_sbert():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return model
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────
INDONESIAN_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
    'pada', 'adalah', 'atau', 'juga', 'dalam', 'tidak', 'akan', 'ada',
    'saya', 'kamu', 'anda', 'ia', 'mereka', 'kami', 'kita', 'bisa',
    'sudah', 'bila', 'jika', 'maka', 'oleh', 'karena', 'apa',
    'bagaimana', 'berapa', 'kapan', 'dimana', 'siapa', 'apakah', 'cara',
    'lebih', 'sangat', 'dapat', 'nya', 'pun', 'lagi', 'belum',
    'telah', 'namun', 'tapi', 'serta', 'meski', 'agar', 'supaya', 'hal',
    'the', 'is', 'are', 'was', 'what', 'how', 'why', 'when', 'where'
}


@st.cache_resource
def get_preprocessor():
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
    all_stopwords = INDONESIAN_STOPWORDS | english_stopwords

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in all_stopwords and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    return preprocess_text


# ──────────────────────────────────────────────────────────────────────────
# CHATBOT ENGINE
# ──────────────────────────────────────────────────────────────────────────
class MedicalChatbotEngine:
    """Hybrid retrieval chatbot: rule-based intent + SBERT/TF-IDF semantic search."""

    def __init__(self, dataframe, preprocess_fn, sbert_model=None, threshold=0.35, top_k=3):
        self.df = dataframe.reset_index(drop=True)
        self.preprocess = preprocess_fn
        self.sbert_model = sbert_model
        self.use_sbert = sbert_model is not None
        self.threshold = threshold if self.use_sbert else 0.15
        self.top_k = min(top_k, len(self.df))
        self.conversation_history = []
        self._build_index()
        self._define_rules()

    def _build_index(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if 'processed_question' not in self.df.columns:
            self.df['processed_question'] = self.df['question'].apply(self.preprocess)

        if self.use_sbert:
            self.sbert_embeddings = self.sbert_model.encode(
                self.df['question'].tolist(), convert_to_tensor=True
            )

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=5000, sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_question'])

    def _define_rules(self):
        self.rules = {
            'emergency': {
                'patterns': [
                    r'(sesak.*berat|nyeri dada.*berat|tidak.*bernapas|pingsan|'
                    r'tidak sadarkan diri|kejang|stroke|wajah.*lumpuh)'
                ],
                'responses': [
                    "🚨 **DARURAT!** Ini terdengar seperti kondisi gawat darurat. "
                    "Segera hubungi **119** atau ke **IGD terdekat**. Jangan ditunda!"
                ],
            },
            'greeting': {
                'patterns': [r'\b(halo|hai|hi|hello|selamat pagi|selamat siang|selamat malam)\b'],
                'responses': ["👋 Halo! Saya MedBot. Ada keluhan atau pertanyaan kesehatan yang bisa saya bantu?"],
            },
            'capability': {
                'patterns': [r'(topik apa|bisa apa|kamu bisa|kemampuan)'],
                'responses': [None],  # ditangani khusus di get_response
            },
        }

    def _check_rules(self, text):
        text_l = text.lower()
        for intent, data in self.rules.items():
            for pattern in data['patterns']:
                if re.search(pattern, text_l):
                    if intent == 'capability':
                        cats = ', '.join(sorted(self.df['category'].unique()))
                        return f"📋 Topik yang saya kuasai: {cats}."
                    return random.choice(data['responses'])
        return None

    def _search_sbert(self, query):
        from sentence_transformers import util
        emb = self.sbert_model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(emb, self.sbert_embeddings)[0].cpu().numpy()
        top = np.argsort(-scores)[:self.top_k]
        return [(int(i), float(scores[i])) for i in top]

    def _search_tfidf(self, query):
        from sklearn.metrics.pairwise import cosine_similarity
        processed = self.preprocess(query)
        vec = self.vectorizer.transform([processed])
        scores = cosine_similarity(vec, self.tfidf_matrix).flatten()
        top = np.argsort(scores)[::-1][:self.top_k]
        return [(int(i), float(scores[i])) for i in top]

    def _build_context_query(self, user_input):
        if self.conversation_history:
            return self.conversation_history[-1] + " " + user_input
        return user_input

    def get_response(self, user_input):
        if not user_input.strip():
            return "Silakan ketik pertanyaan Anda.", None, None

        rule = self._check_rules(user_input)
        if rule:
            return rule, None, None

        query = self._build_context_query(user_input)

        if self.use_sbert:
            results = self._search_sbert(query)
            method = "SBERT"
        else:
            results = self._search_tfidf(query)
            method = "TF-IDF"

        best_idx, best_score = results[0]
        if best_score < self.threshold:
            self.conversation_history.append(user_input)
            return ("🤔 Maaf, saya belum menemukan jawaban yang cukup relevan untuk "
                    "pertanyaan itu. Coba gunakan kata kunci lain, atau konsultasikan "
                    "langsung ke tenaga medis."), None, None

        boosted = []
        for idx, score in results:
            text = self.df.iloc[idx]['question']
            bonus = sum(1 for w in user_input.lower().split() if w in text.lower())
            boosted.append((idx, score + 0.05 * bonus))
        best_idx = sorted(boosted, key=lambda x: x[1], reverse=True)[0][0]

        row = self.df.iloc[best_idx]
        self.conversation_history.append(user_input)
        return row['answer'], row['category'], method


# ──────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset...")
def load_data(file_or_path):
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {'category', 'question', 'answer'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {missing}. "
                          f"Kolom yang ada: {list(df.columns)}")
    df = df.dropna(subset=['question', 'answer']).drop_duplicates(subset=['question'])
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Membangun index pencarian chatbot...")
def build_engine(df, use_sbert_toggle):
    preprocess_fn = get_preprocessor()
    sbert_model = load_sbert() if use_sbert_toggle else None
    return MedicalChatbotEngine(df, preprocess_fn, sbert_model=sbert_model)


# ──────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 MedBot")
    st.caption("Chatbot NLP kesehatan berbasis retrieval semantic search.")

    uploaded_file = None
    if not os.path.exists(DATA_PATH):
        st.warning(f"File `{DATA_PATH}` tidak ditemukan di repo.")
        uploaded_file = st.file_uploader("Upload dataset CSV (question, answer, category)", type=["csv"])

    use_sbert_toggle = st.checkbox(
        "Gunakan SBERT (semantic search)",
        value=True,
        help="Matikan untuk pakai TF-IDF saja — lebih ringan & cepat di server gratis.",
    )

    st.divider()
    if st.button("🗑️ Reset percakapan"):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA & BUILD ENGINE
# ──────────────────────────────────────────────────────────────────────────
data_source = DATA_PATH if os.path.exists(DATA_PATH) else uploaded_file

if data_source is None:
    st.title("🩺 MedBot - Asisten Kesehatan")
    st.info("👈 Upload file dataset CSV di sidebar untuk memulai (kolom: category, question, answer).")
    st.stop()

try:
    df = load_data(data_source)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

engine = build_engine(df, use_sbert_toggle)

with st.sidebar:
    st.metric("Total data Q&A", len(df))
    st.metric("Jumlah kategori", df['category'].nunique())
    st.metric("Mode aktif", "SBERT" if engine.use_sbert else "TF-IDF")
    with st.expander("Lihat kategori"):
        st.write(", ".join(sorted(df['category'].unique())))

# ──────────────────────────────────────────────────────────────────────────
# MAIN CHAT UI
# ──────────────────────────────────────────────────────────────────────────
st.title("🩺 MedBot - Asisten Kesehatan")
st.caption("Tanyakan gejala, penyebab, atau cara penanganan awal kondisi kesehatan umum.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Halo! Saya MedBot. Ada keluhan atau pertanyaan kesehatan yang bisa saya bantu?",
         "category": None, "method": None}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("category"):
            st.markdown(f"<span class='medbot-badge'>{msg['category']} · {msg['method']}</span>", unsafe_allow_html=True)
        st.markdown(msg["content"])

user_input = st.chat_input("Ketik pertanyaan kesehatan Anda di sini...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "category": None, "method": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            answer, category, method = engine.get_response(user_input)
        if category:
            st.markdown(f"<span class='medbot-badge'>{category} · {method}</span>", unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown(
            "<div class='medbot-disclaimer'>⚠️ MedBot bukan pengganti diagnosis dokter. "
            "Untuk kondisi serius, segera konsultasikan ke tenaga medis profesional.</div>",
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "category": category, "method": method}
    )
