from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

EMBED_MODEL = "mistral:latest"
UNIVERSITY_FAISS_PATH = "./university_faiss_index"

query = "What are the effects of Uranus' conjunction with the seven planets?"
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)
db = FAISS.load_local(UNIVERSITY_FAISS_PATH, embedding_model)

results = db.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"\n🔍 Result {i}")
    print("📄 Source:", doc.metadata.get("source", "unknown"))
    print("📘 Content Preview:", doc.page_content[:500])