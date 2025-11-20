# import os, faiss, numpy as np
# from openai import OpenAI
# from .config import INDEX_DIR, TOP_K, CHAT_MODEL, EMBED_MODEL


# client = OpenAI()


# def _load():
#     idx = faiss.read_index(os.path.join(INDEX_DIR,"faiss.index"))
#     corpus = np.load(os.path.join(INDEX_DIR,"corpus.npy"), allow_pickle=True)
#     return idx, corpus


# def _embed(q: str) -> np.ndarray:
#     r = client.embeddings.create(model=EMBED_MODEL, input=[q])
#     v = np.array(r.data[0].embedding, dtype="float32").reshape(1,-1)
#     faiss.normalize_L2(v); 
#     return v


# def retrieve(question: str, k:int=TOP_K):
#     idx, corpus = _load()
#     v = _embed(question)
#     D,I = idx.search(v, min(k, len(corpus)))
#     return [str(corpus[i]) for i in I[0]]


# def answer_with_llm(question: str, chunks):
#     ctx = "\n\n---\n\n".join(chunks)
#     sys = ("Tu es un assistant FR/EN. Réponds UNIQUEMENT à partir du contexte. "
#            "Si ça manque, dis: \"Je ne sais pas d'après le contexte fourni.\"")
#     user = f"Question:\n{question}\n\nContexte:\n{ctx}\n\nRéponse:"
#     r = client.chat.completions.create(
#         model=CHAT_MODEL, temperature=0.2, max_tokens=300,
#         messages=[{"role":"system","content":sys},{"role":"user","content":user}]
#     )
#     return r.choices[0].message.content.strip()


# def build_answer_local(question, chunks):
#     if not chunks:
#         return "Je ne sais pas d'après le contexte fourni."
#     best = chunks[0].strip().replace("\n"," ")
#     return "Voici l'information trouvée dans les documents :\n" + best


# def answer(question: str, k: int = TOP_K):
#     chunks = retrieve(question, k=k)
#     try:
#         return answer_with_llm(question, chunks)
#     except Exception:
#         return build_answer_local(question, chunks)
    


# if __name__=="__main__":
#     q="Quelles sont vos heures d'ouverture ?"
#     print( answer(q, k=3))


import os, faiss, numpy as np
from openai import OpenAI
from .config import INDEX_DIR, TOP_K, CHAT_MODEL, EMBED_MODEL


# for k in ["OPENAI_BASE_URL","OPENAI_API_BASE","OPENAI_ORG","OPENAI_PROJECT"]:
#     print(k, os.getenv(k))

client = OpenAI()

# print("client ok")
# liste vite un modèle public
# print([m.id for m in client.models.list().data[:3]])

def _load():
    idx = faiss.read_index(os.path.join(INDEX_DIR,"faiss.index"))
    corpus = np.load(os.path.join(INDEX_DIR,"corpus.npy"), allow_pickle=True)
    return idx, corpus

def _embed(q: str) -> np.ndarray:
    r = client.embeddings.create(model=EMBED_MODEL, input=[q])
    print("EMB TOKENS:", r.usage)
    v = np.array(r.data[0].embedding, dtype="float32").reshape(1,-1)
    faiss.normalize_L2(v); 
    return v

def retrieve(question: str, k:int=TOP_K):
    idx, corpus = _load()
    v = _embed(question)
    D,I = idx.search(v, min(k, len(corpus)))
    return [str(corpus[i]) for i in I[0]]

def answer_with_llm(question: str, chunks):
    ctx = "\n\n---\n\n".join(chunks)
    sys = ("Tu es un assistant FR/EN. Réponds UNIQUEMENT à partir du contexte. "
           "Si ça manque, dis: \"Je ne sais pas d'après le contexte fourni.\"")
    user = f"Question:\n{question}\n\nContexte:\n{ctx}\n\nRéponse:"
    r = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2, max_tokens=300,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    print("TOKENS:", r.usage)
    return r.choices[0].message.content.strip()


def build_answer_local(question, chunks):
     if not chunks:
         return "Je ne sais pas répondre d'après le contexte fourni."
     best = chunks[0].strip().replace("\n"," ")
     return "Voici l'information trouvée dans les documents :\n" + best


def answer(question: str, k: int = TOP_K):
     chunks = retrieve(question, k=k)
     try:
         return answer_with_llm(question, chunks)
     except Exception:
         return build_answer_local(question, chunks)


if __name__=="__main__":
    q1="Quelles sont vos heures d'ouverture ?"
    print("Answer 1 : ", answer(q1, k=3))

    q2="Que faire en cas d'erreur de paiement ?"
    print("Answer 2 : ", answer(q2, k=3))

    q3="Quels sont mes droits ?"
    print("Answer 3 : ", answer(q3, k=3))

    q4="Comment contacter le support technique ?"
    print("Answer 4 : ", answer(q4, k=3))

    q5="Comment contacter le support technique ?"
    print("Answer 5 : ", answer(q5, k=3))

    q6="Combien de temps ai-je pour retourner un produit ?"
    print("Answer 6 : ", answer(q6, k=3))

    q7="Coment crééer et activer mon compte ?"
    print("Answer 7 : ", answer(q7, k=3))

