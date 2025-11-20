import os, numpy as np, faiss
from typing import List
from openai import OpenAI

from .config import DOCS_DIR, INDEX_DIR, EMBED_MODEL
from .pdf_utils import extract_text_from_pdf

client = OpenAI()

def load_docs() -> List[str]:

    texts=[]

    for fn in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, fn)
        if not os.path.isfile(path):
            continue

        if fn.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                t=f.read().strip()
                if t: 
                    texts.append(t)

        elif fn.lower().endswith(".pdf"):
            t = extract_text_from_pdf(path)
            if t : 
                texts.append(t)

    return texts


def embed(texts: List[str]) -> np.ndarray:

    # Calcule les embeddings OpenAI pour une liste de textes.
    # Retourne un tableau numpy (n_docs, dim).
    
    vecs=[]; 
    batch_size=64

    for i in range(0,len(texts),batch_size):
        r=client.embeddings.create(model=EMBED_MODEL, input=texts[i:i+batch_size])
        vecs.extend([d.embedding for d in r.data])

    return np.array(vecs, dtype="float32")


if __name__=="__main__":
    os.makedirs(INDEX_DIR, exist_ok=True)

    corpus = load_docs()
    if not corpus: 
        raise SystemExit("Aucun .txt dans data/docs")
    
    print(f"[embed_build] Nombre de documents chargés : {len(corpus)}")

    X = embed(corpus)
    print(f"[embed_build] Shape des embeddings : {X.shape}")


    faiss.normalize_L2(X)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)

    faiss.write_index(idx, os.path.join(INDEX_DIR,"faiss.index"))
    np.save(os.path.join(INDEX_DIR,"corpus.npy"), np.array(corpus, dtype=object))

    print("Index créé avec", len(corpus), "documents.")