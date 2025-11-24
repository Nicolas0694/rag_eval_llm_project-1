Projet de démonstration d’un système **RAG + LLM** pour répondre à des questions de support client à partir d’un corpus de documents (TXT + PDF), avec un pipeline d’évaluation simple mais structuré.

Ce projet montre comment :

- Construire un index vectoriel à partir de documents locaux.
- Interroger cet index avec des embeddings OpenAI.
- Générer des réponses avec un LLM conditionné par le contexte.
- Évaluer automatiquement la qualité des réponses via un dataset de questions / faits attendus.

---

## 1. Structure du projet

```text
rag-eval-llm/
├─ .venv/                  # environnement virtuel Python (local)
├─ .env                    # variables d'environnement (clé API, modèles)
├─ requirements.txt        # dépendances Python
├─ data/
│  ├─ docs/                # documents source (.txt + .pdf)
│  │  ├─ horaires.txt
│  │  ├─ retours.txt
│  │  ├─ conditions_generales.txt
│  │  ├─ faq_clients.txt
│  │  ├─ support_technique.txt
│  │  ├─ politique_confidentialite.txt
│  │  └─ guide_utilisateur.pdf
│  ├─ index/               # index FAISS + corpus embarqué
│  │  ├─ faiss.index
│  │  └─ corpus.npy
│  ├─ evals/
│  │  └─ dataset.csv       # dataset d’évaluation (questions + expected_facts)
│  └─ results/
│     └─ eval_report.json  # rapport d’évaluation
└─ src/
   ├─ __init__.py
   ├─ config.py            # chemins, modèles, TOP_K, etc.
   ├─ pdf_utils.py         # extraction de texte depuis les PDF
   ├─ embed_build.py       # build de l’index embeddings + FAISS
   ├─ rag_llm.py           # RAG + génération via LLM
   ├─ eval_simple.py       # évaluation basée sur expected_facts
   └─ chat_demo.py         # petit test direct de l’API chat
```
python -m src pip install openai, faiss-cpu, numpy, PyPDF2, python-dotenv

## 2. Prérequis

- Python 3.10+ recommandé
- Un compte OpenAI avec :
    - une clé API valide
    - accès aux modèles :
        - embeddings : text-embedding-3-small (par exemple)
        - chat : gpt-4o-mini (ou équivalent)


## 3. Installation
  3.1. Cloner le dépôt
        **git clone https://github.com/<ton-compte>/rag-eval-llm.git**
        **cd rag-eval-llm**

  3.2. Créer et activer l’environnement virtuel

      Sur Windows (PowerShell) :
          ** python -m venv .venv **
          ** .\.venv\Scripts\Activate.ps1 **


      Sur macOS / Linux :

          ** python -m venv .venv **
          ** source .venv/bin/activate **

3.3. Installer les dépendances

** python -m pip install --upgrade pip **
** python -m pip install -r requirements.txt **



Si tu reconstruis le projet from scratch, les principales dépendances sont :

** python -m pip install openai python-dotenv faiss-cpu numpy scikit-learn PyPDF2 **
** python -m pip freeze > requirements.txt **

4. Configuration

Créer un fichier .env à la racine du projet :

OPENAI_API_KEY=sk-...             # ta clé API
OPENAI_MODEL=gpt-4o-mini          # modèle de chat
OPENAI_EMBED=text-embedding-3-small


Ces variables sont lues dans src/config.py.

5. Usage

  5.1. Construire l’index (embeddings + FAISS)

  Après avoir déposé tes documents dans data/docs/ :

  ** python -m src.embed_build **

  Ce script :

  - lit tous les .txt et .pdf de data/docs/,

  - appelle l’API d’embeddings OpenAI,

  - normalise les vecteurs,

  - construit un index FAISS,

sauvegarde :

- data/index/faiss.index

- data/index/corpus.npy

5.2. Tester le LLM seul

Petit test de l’API chat :

** python -m src.chat_demo **


Tu dois voir s’afficher une phrase de réponse.

5.3. Tester le RAG (question → réponse)

** python -m src.rag_llm **

Par défaut, le script pose une question de test :

“Quelles sont vos heures d’ouverture ?”

Il :

- embed la question,

- récupère les documents les plus proches via FAISS,

- construit un prompt avec le contexte,

- appelle le LLM,

- affiche la réponse.

6. Évaluation

L’évaluation est basée sur :

data/evals/dataset.csv :

  - id,question,expected_facts,k

  - expected_facts = liste de petits substrings séparés par |

src/eval_simple.py :

exécute le pipeline RAG + LLM sur chaque question,

mesure :

- recall_questions (questions avec ≥ 60 % des faits corrects)

- avg_fact_ratio (couverture moyenne des faits)

P50 / P95 de latence

Lancer l’éval
** python -m src.eval_simple ** 


Le script affiche un résumé, par exemple :

{
  "n": 40,
  "recall_questions": 0.85,
  "recall_questions_pct": "85.0%",
  "avg_fact_ratio": 0.835,
  "avg_fact_ratio_pct": "83.5%",
  "p50_ms": 2214,
  "p95_ms": 5947,
  "threshold_fact_ok": 0.6,
  "out": "data/results/eval_report.json"
}


Le détail question par question est disponible dans data/results/eval_report.json.

7. Résultats actuels (baseline)

Sur le dataset d’évaluation v1.0 (40 questions FR/EN variées), avec :

- RAG embeddings OpenAI + FAISS

- LLM gpt-4o-mini (ou équivalent)

- TOP_K = 5

prompt système demandant d’énumérer explicitement les règles/conditions/droits

on obtient :

- Recall_questions ≈ 85 %

- Avg_fact_ratio ≈ 83,5 %

- P50 ≈ 2,2 s, P95 ≈ 5,9 s

Ces valeurs servent de baseline pour toute amélioration future du système (retrieval, prompts, modèles).


8. Améliorations possibles


- Reranking des documents (BM25 ou autre) avant envoi au LLM.

- Meilleure segmentation des documents longs (chunking plus fin).

- Ajout de métriques plus avancées (similarité d’embeddings réponse / référence, LLM-as-judge).

- Extension à d’autres domaines ou à des documents semi-structurés (CSV, HTML, etc.).
