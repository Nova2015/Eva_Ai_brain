"""
EVA Chroma स्कॅनर (सुरक्षित: प्रत्येक DB ची temp copy वर काम करतो, मूळ फाइल्सना स्पर्श नाही).

चालवा (.venv311 सक्रिय असताना):
    python eva_scan_chroma.py
    python eva_scan_chroma.py "D:\\path\\chroma_db1" "D:\\path\\chroma_db2"
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULTS = [
    r"D:\Eva_Ai_brain\src\rag\data\chroma_db",
    r"D:\Eva_Ai_brain\eva-ml-service\data\chroma_db",
    r"D:\Rag project\eva-ml-service\data\chroma_db",
    r"D:\Rag project\src\rag\data\chroma_db",
]
KEYWORDS = ["पाडगावकर", "भारत देशा", "जय जय"]
DEVA = re.compile(r"[\u0900-\u097F]")
MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def cos_rows(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (an * bn).sum(axis=1)


def scan(path, models):
    print("\n" + "=" * 70)
    print("DB:", path)
    p = Path(path)
    if not (p / "chroma.sqlite3").exists():
        print("  chroma.sqlite3 नाही — वगळले")
        return
    tmp = Path(tempfile.mkdtemp(prefix="evascan_"))
    work = tmp / "db"
    try:
        shutil.copytree(p, work)
        client = chromadb.PersistentClient(path=str(work), settings=Settings(anonymized_telemetry=False))
        cols = [getattr(c, "name", str(c)) for c in client.list_collections()]
        if not cols:
            print("  collections: नाहीत (रिकामा)")
            return
        for name in cols:
            col = client.get_collection(name)
            n = col.count()
            print(f"\n  collection '{name}': count={n}")
            if n == 0:
                continue
            data = col.get(include=["documents", "metadatas"])
            docs = [t or "" for t in data["documents"]]
            metas = data["metadatas"] or []
            print("   workspace_id:", Counter((m or {}).get("workspace_id") for m in metas).most_common(4))
            print("   source:", Counter((m or {}).get("source") for m in metas).most_common(5))
            deva = sum(1 for t in docs if DEVA.search(t))
            print(f"   देवनागरी असलेले chunks: {deva}")
            for kw in KEYWORDS:
                print(f"   '{kw}' असलेले chunks: {sum(1 for t in docs if kw in t)}")

            samp = col.get(limit=8, include=["documents", "embeddings"])
            sd = [t or "" for t in samp["documents"]]
            se = np.array(samp["embeddings"], dtype=np.float32)
            print(f"   stored dim: {se.shape[1]}")
            for mname, model in models.items():
                new = model.encode(sd, show_progress_bar=False)
                if new.shape[1] != se.shape[1]:
                    print(f"   {mname.split('/')[-1]}: dim वेगळा")
                    continue
                print(f"   {mname.split('/')[-1]}: सरासरी cosine (stored vs re-encoded) = {cos_rows(new, se).mean():.3f}")
    except Exception as e:
        print("  त्रुटी:", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    paths = sys.argv[1:] or DEFAULTS
    models = {m: SentenceTransformer(m) for m in MODEL_NAMES}
    for pth in paths:
        scan(pth, models)


if __name__ == "__main__":
    main()
