import csv
import json
import math
import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector-search")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="SearchTech Solutions - Vector Search + Re-ranking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------

DOCUMENTS = []          # list of dicts: doc_id, title, department, year, region, text
EMBEDDINGS = {}          # doc_id -> list[float]
RERANKER_SCORES = {}     # query_id -> {doc_id: score}


def _coerce(value: str):
    """documents.csv has 'year' as int and everything else as string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_data():
    global DOCUMENTS, EMBEDDINGS, RERANKER_SCORES

    docs_path = os.path.join(BASE_DIR, "documents.csv")
    with open(docs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        docs = []
        for row in reader:
            row = dict(row)
            if "year" in row:
                row["year"] = _coerce(row["year"])
            docs.append(row)
        DOCUMENTS = docs

    with open(os.path.join(BASE_DIR, "embeddings.json"), encoding="utf-8") as f:
        EMBEDDINGS = json.load(f)

    with open(os.path.join(BASE_DIR, "reranker_scores.json"), encoding="utf-8") as f:
        RERANKER_SCORES = json.load(f)

    logger.info("Loaded %d documents, %d embeddings, %d reranker score sets",
                len(DOCUMENTS), len(EMBEDDINGS), len(RERANKER_SCORES))


load_data()

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def matches_condition(doc_value, condition) -> bool:
    if isinstance(condition, dict):
        for op, target in condition.items():
            if op == "gte":
                if doc_value is None or not (doc_value >= target):
                    return False
            elif op == "lte":
                if doc_value is None or not (doc_value <= target):
                    return False
            elif op == "gt":
                if doc_value is None or not (doc_value > target):
                    return False
            elif op == "lt":
                if doc_value is None or not (doc_value < target):
                    return False
            elif op == "in":
                if doc_value not in target:
                    return False
            elif op == "eq":
                if doc_value != target:
                    return False
            else:
                # unknown operator - fail closed (no match) rather than
                # silently accepting everything
                return False
        return True
    else:
        # exact match
        return doc_value == condition


def apply_filter(documents, filter_dict):
    if not filter_dict:
        return list(documents)
    filtered = []
    for doc in documents:
        ok = True
        for field, condition in filter_dict.items():
            doc_value = doc.get(field)
            if not matches_condition(doc_value, condition):
                ok = False
                break
        if ok:
            filtered.append(doc)
    return filtered


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def vector_search(query_id, query_vector, top_k, rerank_top_n, filter_dict):
    # Stage 1: filter + cosine similarity
    candidates = apply_filter(DOCUMENTS, filter_dict)

    scored = []
    for doc in candidates:
        doc_id = doc["doc_id"]
        emb = EMBEDDINGS.get(doc_id)
        if emb is None:
            continue
        sim = cosine_similarity(query_vector, emb)
        scored.append((doc_id, sim))

    # sort desc by similarity, tie-break lexicographically smaller doc_id
    scored.sort(key=lambda t: (-t[1], t[0]))
    top_k_docs = [doc_id for doc_id, _ in scored[:top_k]]

    # Stage 2: re-rank using precomputed reranker scores
    query_scores = RERANKER_SCORES.get(query_id, {})
    reranked = []
    for doc_id in top_k_docs:
        score = query_scores.get(doc_id, 0.0)
        reranked.append((doc_id, score))

    reranked.sort(key=lambda t: (-t[1], t[0]))
    final = [doc_id for doc_id, _ in reranked[:rerank_top_n]]
    return final


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/vector-search")
async def vector_search_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"matches": []})

    if not isinstance(body, dict):
        return JSONResponse(content={"matches": []})

    query_id = body.get("query_id")
    query_vector = body.get("query_vector")
    top_k = body.get("top_k")
    rerank_top_n = body.get("rerank_top_n")
    filter_dict = body.get("filter") or {}

    # Graceful handling of malformed / missing inputs
    if not query_id or not isinstance(query_vector, list) or not query_vector:
        return JSONResponse(content={"matches": []})

    try:
        top_k = int(top_k) if top_k is not None else len(DOCUMENTS)
    except (TypeError, ValueError):
        top_k = len(DOCUMENTS)

    try:
        rerank_top_n = int(rerank_top_n) if rerank_top_n is not None else top_k
    except (TypeError, ValueError):
        rerank_top_n = top_k

    top_k = max(0, top_k)
    rerank_top_n = max(0, rerank_top_n)

    if not isinstance(filter_dict, dict):
        filter_dict = {}

    try:
        query_vector = [float(x) for x in query_vector]
    except (TypeError, ValueError):
        return JSONResponse(content={"matches": []})

    try:
        matches = vector_search(query_id, query_vector, top_k, rerank_top_n, filter_dict)
    except Exception as e:
        logger.exception("vector_search failed: %s", e)
        matches = []

    return JSONResponse(content={"matches": matches})


@app.get("/")
async def health():
    return {"status": "ok", "service": "vector-search-api", "documents_loaded": len(DOCUMENTS)}
