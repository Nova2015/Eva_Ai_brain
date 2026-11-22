"""
EVA embedding diagnostic (फक्त वाचन — Supabase ला स्पर्श नाही).

चालवा (.venv311 सक्रिय असताना):
    python eva_embed_check.py "D:\\Eva_Ai_brain\\data\\chroma_marathi_test_copy"
(मूळ फोल्डरची copy वापरा; Chroma उघडताना फाइल्स किरकोळ बदलू शकतात.)
"""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if len(sys.argv) < 2:
    sys.exit("Usage: python eva_embed_check.py <chroma_folder>")

CHROMA_DIR = Path(sys.argv[1])
WANT = os.getenv("CHROMA_COLLECTION", "eva_knowledge")
MAX_DOCS = 800  # मोठ्या collection मध्ये इतकेच chunks तपासतो (keyword-chunks नेहमी समाविष्ट)

QUERIES = [
    ("जय जय हे भारत देशा मंगेश पाडगावकर", "भारत देशा"),
]

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    ("intfloat/multilingual-e5-small", "query: ", "passage: "),
]


def cos_matrix(q, d):
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return qn @ dn.T


def main():
    print(f"Chroma path: {CHROMA_DIR}  exists={CHROMA_DIR.exists()}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    cols = [(getattr(c, "name", str(c))) for c in client.list_collections()]
    infos = [(n, client.get_collection(n).count()) for n in cols]
    print("Collections (नाव, count):", infos)
    if not infos:
        sys.exit("collection नाही — हा चुकीचा फोल्डर आहे.")

    name = WANT if WANT in cols else max(infos, key=lambda x: x[1])[0]
    col = client.get_collection(name)
    print(f"वापरत आहे: '{name}'  count={col.count()}")

    data = col.get(include=["documents", "embeddings", "metadatas"])
    docs = [t or "" for t in data["documents"]]
    stored = np.array(data["embeddings"], dtype=np.float32)
    metas = data["metadatas"] or []
    print(f"Stored embedding dim: {stored.shape[1]}")
    print("workspace_id:", Counter((m or {}).get("workspace_id") for m in metas).most_common(5))
    print("source:", Counter((m or {}).get("source") for m in metas).most_common(5))

    for query, keyword in QUERIES:
        hit_all = [i for i, t in enumerate(docs) if keyword in t]
        print("\n" + "=" * 70)
        print(f"प्रश्न: {query}")
        print(f"'{keyword}' असलेले chunks: {len(hit_all)}  (index: {hit_all[:10]})")

        if len(docs) > MAX_DOCS:
            step = max(1, len(docs) // MAX_DOCS)
            idx = sorted(set(hit_all) | set(range(0, len(docs), step)))
            print(f"collection मोठी आहे; {len(idx)} chunks चा नमुना तपासतो")
        else:
            idx = list(range(len(docs)))
        sub_docs = [docs[i] for i in idx]
        sub_stored = stored[idx]
        hit_pos = [p for p, i in enumerate(idx) if i in set(hit_all)]

        for mname, qp, pp in MODELS:
            print("\n--- " + mname)
            try:
                model = SentenceTransformer(mname)
            except Exception as e:
                print("   load failed:", e)
                continue

            q = model.encode([qp + query])
            d_new = model.encode([pp + t for t in sub_docs], batch_size=16, show_progress_bar=False)

            if d_new.shape[1] == sub_stored.shape[1]:
                same = cos_matrix(d_new[:1], sub_stored[:1])[0, 0]
                print(f"   stored vs re-encoded cosine = {same:.4f}  (~1.0 = हेच model index साठी वापरले)")
            else:
                print(f"   dim वेगळा ({d_new.shape[1]} vs stored {sub_stored.shape[1]})")

            sims = cos_matrix(q, d_new)[0]
            order = np.argsort(-sims)
            for rank, p in enumerate(order[:3], 1):
                prev = sub_docs[p].replace("\n", " ")[:90]
                print(f"   #{rank} score={sims[p]:.3f} idx={idx[p]}  {prev}")
            if hit_pos:
                ranks = [int(np.where(order == p)[0][0]) + 1 for p in hit_pos]
                print(f"   keyword-chunk चा सर्वोत्तम rank: {min(ranks)} / {len(sub_docs)}")


if __name__ == "__main__":
    main()
