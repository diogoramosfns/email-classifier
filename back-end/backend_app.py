"""
Backend Flask app to classify emails as 'Produtivo' or 'Improdutivo' and to suggest an automatic reply.

Features:
- Accepts .txt or .pdf uploads or raw email text via JSON/form.
- Preprocesses text (lowercasing, punctuation removal, stopword removal, stemming).
- Uses a small built-in sklearn pipeline (TF-IDF + LogisticRegression) with example training data as a fallback.
- If OPENAI_API_KEY is set, will call OpenAI's chat completion to optionally refine classification and to generate a suggested reply.
- Endpoints:
  - POST /process -> accepts file or text, returns JSON {category, confidence, suggested_response}
  - POST /train   -> (optional) upload CSV with columns 'text' and 'label' to retrain the local classifier

Notes:
- This is a ready-to-run prototype. For production you should:
  - secure the OpenAI key and endpoints
  - add authentication & rate limiting
  - run model training offline and persist the model (joblib)
  - add logging and monitoring

Run:
venv\\Scripts\\activate
python backend_app.py
  
"""

from dotenv import load_dotenv

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import re
import json

# NLP / ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# PDF parsing
from PyPDF2 import PdfReader

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

# Ensure NLTK resources
nltk_packages = ["punkt", "stopwords", "punkt_tab"]
for pkg in nltk_packages:
    try:
        if pkg == "stopwords":
            nltk.data.find("corpora/stopwords")
        elif pkg == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")
        else:
            nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download(pkg, quiet=False)

# Stopwords com fallback
try:
    STOPWORDS = set(stopwords.words("portuguese"))
except LookupError:
    STOPWORDS = set()
    print("[WARN] Stopwords em português não encontradas. Nenhuma será removida.")

STEMMER = SnowballStemmer("portuguese")

MODEL_PATH = "email_classifier.joblib"

app = Flask(__name__)
CORS(app)

# --- Utilities ---
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^\w\sáéíóúâêôãõçàèì]+", " ", text, flags=re.UNICODE)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)

# --- Simple fallback training dataset ---
DEFAULT_DATA = [
    ("Olá, preciso de uma atualização sobre o processo X. Está pendente desde a última semana.", "Produtivo"),
    ("Bom dia, gostaria de saber o status do pedido #12345.", "Produtivo"),
    ("Parabéns a toda a equipe pelo excelente trabalho neste ano! Boas festas.", "Improdutivo"),
    ("Obrigado pelo suporte, tudo certo.", "Improdutivo"),
    ("Encaminho em anexo o comprovante de pagamento.", "Produtivo"),
    ("Feliz natal e próspero ano novo.", "Improdutivo"),
    ("Preciso que alguém me retorne com a previsão de entrega.", "Produtivo"),
    ("Apenas agradecendo pelo envio anterior.", "Improdutivo"),
]

def build_and_train_pipeline(dataset=None):
    if dataset is None:
        dataset = DEFAULT_DATA
    texts = [preprocess_text(t) for t, _ in dataset]
    labels = [l for _, l in dataset]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

def load_pipeline():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return build_and_train_pipeline()
    else:
        return build_and_train_pipeline()

PIPELINE = load_pipeline()

def classify_local(text: str):
    ptext = preprocess_text(text)
    probs = PIPELINE.predict_proba([ptext])[0]
    classes = PIPELINE.classes_
    idx = probs.argmax()
    label = classes[idx]
    confidence = float(probs[idx])
    return label, confidence

# --- OpenAI integration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and openai is not None:
    openai.api_key = OPENAI_API_KEY

def classify_with_openai(text: str):
    if openai is None or OPENAI_API_KEY is None:
        raise RuntimeError("OpenAI library or API key not available")

    system = (
        "Você é um classificador para emails corporativos. "
        "Classifique o email em uma das duas categorias: 'Produtivo' ou 'Improdutivo'. "
        "Retorne um JSON com as chaves: category, confidence, suggested_response."
    )
    prompt = f"Email:\n{text}\n\nResponda apenas com um JSON válido."

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.0,
    )
    content = resp["choices"][0]["message"]["content"]

    try:
        j = json.loads(content)
        return j.get("category"), float(j.get("confidence", 0)), j.get("suggested_response")
    except Exception:
        return None, 0.0, None

def generate_reply_with_openai(category: str, text: str):
    if openai is None or OPENAI_API_KEY is None:
        return None

    system = "Você é um assistente que escreve respostas profissionais e concisas para emails corporativos em português brasileiro."
    user_prompt = (
        f"O email recebido é:\n{text}\n\nA categoria detectada foi '{category}'. "
        "Escreva uma resposta profissional, curta (2-4 frases), adequada à categoria."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=200,
        temperature=0.3,
    )

    return resp["choices"][0]["message"]["content"].strip()

# --- Flask routes ---
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/process", methods=["POST"])
def process_email():
    text = None
    if "text" in request.form:
        text = request.form.get("text")
    elif request.is_json:
        data = request.get_json()
        text = data.get("text") if isinstance(data, dict) else None

    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename.lower()
        if filename.endswith('.pdf'):
            buf = io.BytesIO(f.read())
            try:
                text = extract_text_from_pdf(buf)
            except Exception as e:
                return jsonify({"error": "Não foi possível ler o PDF: " + str(e)}), 400
        else:
            text = f.read().decode('utf-8', errors='ignore')

    if not text or text.strip() == "":
        return jsonify({"error": "Nenhum texto recebido"}), 400

    label, confidence = classify_local(text)
    
    suggested_response = None
    used_openai = False
    if confidence < 0.75 and OPENAI_API_KEY and openai is not None:
        try:
            o_cat, o_conf, o_resp = classify_with_openai(text)
            if o_cat:
                label = o_cat
                confidence = max(confidence, o_conf)
            if o_resp:
                suggested_response = o_resp
            used_openai = True
        except Exception:
            used_openai = False

    if not suggested_response and OPENAI_API_KEY and openai is not None:
        try:
            suggested_response = generate_reply_with_openai(label, text)
            used_openai = True
        except Exception:
            suggested_response = None

    if not suggested_response:
        if label == "Produtivo":
            suggested_response = (
                "Obrigado pelo contato. Recebemos sua mensagem e estamos verificando. "
                "Em até 48 horas retornaremos com uma atualização ou solicitação de informação adicional."
            )
        else:
            suggested_response = (
                "Agradecemos a mensagem. Desejamos sucesso e estamos à disposição para qualquer necessidade futura."
            )

    return jsonify({
        "category": label,
        "confidence": round(confidence, 4),
        "suggested_response": suggested_response,
        "used_openai": used_openai
    })

@app.route("/train", methods=["POST"])
def train():
    dataset = None
    if "file" in request.files:
        f = request.files["file"]
        content = f.read().decode('utf-8', errors='ignore')
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        parsed = []
        for ln in lines:
            parts = ln.split(',')
            if len(parts) >= 2:
                label = parts[-1].strip().strip('"')
                text = ",".join(parts[:-1]).strip().strip('"')
                parsed.append((text, label))
        if parsed:
            dataset = parsed

    pipeline = build_and_train_pipeline(dataset)
    return jsonify({"status": "trained", "n_samples": len(dataset) if dataset else len(DEFAULT_DATA)})

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        build_and_train_pipeline()
    print("[INFO] - Todos os recursos NLTK foram carregados e o modelo foi inicializado.")
    print(f"[OPEN AI KEY] - {OPENAI_API_KEY}")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
