from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# 설정값
INDEX_PATH = "./university_faiss_index"
QUERY = "What is the 6th house in traditional astrology?"
EMBED_MODEL = "mistral:latest"

# 임베딩 모델 초기화
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# FAISS 인덱스 불러오기
retriever = FAISS.load_local(INDEX_PATH, embeddings).as_retriever()

# 쿼리 실행
docs = retriever.get_relevant_documents(QUERY)

# 결과 출력
for i, d in enumerate(docs, 1):
    print(f"\n🔍 Result {i}")
    print(f"📄 Source: {d.metadata.get('source', 'Unknown')}")
    print(f"📘 Preview: {d.page_content[:300]}")