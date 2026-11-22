"""
EVA मराठी retrieval चाचणी (सुरक्षित: temp copy वर, फक्त वाचन).
चालवा (.venv311 सक्रिय असताना):
    python eva_poem_test.py
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Eva_Ai_brain\src\rag\data\chroma_db")
COLLECTION = "eva_knowledge"
WORKSPACE = "default"  # मराठी txt चे 341 chunks इथे आहेत

# (प्रश्न, chunk मध्ये असू शकणारे शब्द — यापैकी कोणताही चालतो)
QUERIES = [
    ("जय जय हे भारत देशा मंगेश पाडगावकर", ["भारत देशा", "पाडगावकर"]),
    ("जय जय हे भारत देशा", ["भारत देशा"]),
    ("पाडगावकर यांची कविता", ["पाडगावकर"]),
]

# (नाव, query prefix, passage prefix, stored vectors वापरायचे?)
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "", "", True),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", False),
    ("intfloat/multilingual-e5-small", "query: ", "passage: ", False),
]


def cos(q, d):
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return (qn @ dn.T)[0]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="evapoem_"))
    try:
        work = tmp / "db"
        shutil.copytree(DB, work)
        client = chromadb.PersistentClient(path=str(work), settings=Settings(anonymized_telemetry=False))
        col = client.get_collection(COLLECTION)

        sub = col.get(where={"workspace_id": WORKSPACE}, include=["documents", "embeddings", "metadatas"])
        docs = [t or "" for t in sub["documents"]]
        stored = np.array(sub["embeddings"], dtype=np.float32)
        print(f"workspace '{WORKSPACE}': {len(docs)} chunks, dim={stored.shape[1]}")

        loaded = {}
        for name, qp, pp, use_stored in MODELS:
            try:
                m = SentenceTransformer(name)
            except Exception as e:
                print("load failed:", name, e)
                continue
            d = stored if use_stored else m.encode([pp + t for t in docs], batch_size=16, show_progress_bar=False)
            loaded[name] = (m, qp, d)

        for query, kws in QUERIES:
            hit = [i for i, t in enumerate(docs) if any(k in t for k in kws)]
            print("\n" + "=" * 70)
            print(f"प्रश्न: {query}")
            print(f"संबंधित शब्द असलेले chunks: {len(hit)}")
            for name, (m, qp, d) in loaded.items():
                sims = cos(m.encode([qp + query]), d)
                order = np.argsort(-sims)
                print(f"\n--- {name.split('/')[-1]}")
                for r, i in enumerate(order[:3], 1):
                    mark = "  <== शब्द आहे" if i in hit else ""
                    print(f"   #{r} {sims[i]:.3f}  {docs[i].replace(chr(10), ' ')[:80]}{mark}")
                if hit:
                    best = min(int(np.where(order == i)[0][0]) + 1 for i in hit)
                    print(f"   संबंधित chunk चा सर्वोत्तम rank: {best} / {len(docs)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
