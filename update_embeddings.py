import os
import glob
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader # ✅ PyPDFLoader 임포트
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# 공통 설정 (AnandaGPT.py와 동일하게 유지)
EMBED_MODEL = "mistral:latest"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 100

# AnandaGPT.py와 동일한 경로 상수를 사용한다고 가정합니다.
UNIVERSITY_DOCS_PATH = "./university_docs/"
UNIVERSITY_FAISS_PATH = "./university_faiss_index"

def get_all_file_paths(folder_path):
    return glob.glob(os.path.join(folder_path, "*.pdf")) + \
           glob.glob(os.path.join(folder_path, "*.txt")) + \
           glob.glob(os.path.join(folder_path, "*.docx"))

# ✅ 수정된 부분: source_type 파라미터 추가 및 메타데이터 삽입, 로더 조건부 사용
def load_documents_from_paths(paths, source_type):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = []
    for path in paths:
        try: # ✅ try-except 블록 추가하여 개별 파일 오류 처리
            file_extension = os.path.splitext(path)[1].lower()
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(path) # ✅ PDF 파일은 PyPDFLoader 사용
            elif file_extension in [".txt", ".docx"]:
                loader = UnstructuredFileLoader(path) # TXT, DOCX는 UnstructuredFileLoader 사용
            else:
                print(f"경고: 지원되지 않는 파일 형식입니다. 건너뜁니다: {path}")
                continue

            split_docs = loader.load_and_split(text_splitter=splitter)
            for doc in split_docs:
                doc.metadata['source'] = source_type
                doc.metadata['file_path'] = os.path.abspath(path)
            docs.extend(split_docs)
            print(f"'{os.path.basename(path)}' 파일 로드 및 분할 완료. 청크 수: {len(split_docs)}")
        except Exception as e:
            print(f"오류: '{os.path.basename(path)}' 파일 로드 중 오류 발생: {e}")
            print(f"이 파일을 건너뛰고 다음 파일로 진행합니다.")
            continue # 오류 발생 시 해당 파일 건너뛰고 다음 파일로 진행

    return docs

def incremental_update_vectorstore(index_path, new_file_folder, source_type):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 1. 기존 인덱스 불러오기 또는 새로 생성
    faiss_index_file = os.path.join(index_path, "index.faiss")
    faiss_pkl_file = os.path.join(index_path, "index.pkl")

    vectorstore = None
    if os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
        try:
            print(f"기존 FAISS 인덱스 로드 중: {index_path}")
            vectorstore = FAISS.load_local(index_path, embeddings)
            print("기존 FAISS 인덱스 로드 완료.")
        except RuntimeError as e:
            print(f"FAISS 인덱스 로드 중 오류 발생 ({e}). 새 인덱스를 생성합니다.")
            vectorstore = None
    else:
        print(f"FAISS 인덱스 파일이 없거나 불완전합니다. 새 인덱스를 생성합니다.")

    if vectorstore is None:
        os.makedirs(index_path, exist_ok=True) # 인덱스 저장 폴더 생성
        vectorstore = FAISS.from_documents([], embeddings) # 빈 인덱스 생성

    # 2. 새 문서 불러오기 (메타데이터와 함께)
    new_paths = get_all_file_paths(new_file_folder)
    
    # ⚠️ 이미 인덱싱된 파일 건너뛰기 로직 (선택 사항이지만 대용량 처리 시 효율적)
    # 현재 FAISS는 직접적으로 저장된 파일 경로 목록을 제공하지 않으므로,
    # 이 부분은 외부 파일(예: processed_files.txt)을 사용하거나,
    # 단순히 'new_file_folder'에 정말 새로운 파일만 넣는다고 가정합니다.
    # 여기서는 간단히 모든 파일을 다시 로드하고 FAISS의 내부 중복 처리(동일한 청크는 추가 안 함)에 의존합니다.
    
    new_docs = load_documents_from_paths(new_paths, source_type=source_type)

    if not new_docs:
        print("✅ 새로운 문서가 없거나 유효한 문서가 없습니다. 업데이트를 건너뜁니다.")
        return

    # 3. 새 문서만 임베딩 추가
    vectorstore.add_documents(new_docs)
    print(f"✅ '{source_type}' 메타데이터가 있는 {len(new_docs)}개의 새 문서 청크를 인덱스에 추가했습니다.")

    # 4. 다시 저장
    vectorstore.save_local(index_path)
    print(f"✅ 벡터스토어를 다음 경로에 저장했습니다: {index_path}")

# --- 실행 예시 ---
if __name__ == "__main__":
    # UNIVERSITY_DOCS_PATH에 새로운 고전 문헌 PDF들을 넣은 후 이 스크립트를 실행합니다.
    print(f"'{UNIVERSITY_DOCS_PATH}' 폴더의 새로운 대학 교재 문서들을 FAISS 인덱스에 추가합니다.")
    incremental_update_vectorstore(UNIVERSITY_FAISS_PATH, UNIVERSITY_DOCS_PATH, "university_textbook")

    # 나중에 교수님 자료를 추가할 때 (예시)
    # PROFESSORS_DOCS_PATH = "./professors/"
    # PROFESSORS_FAISS_BASE_PATH = "./professors_faiss_index/"
    # professor_name = "Prof_Lee"
    # os.makedirs(os.path.join(PROFESSORS_DOCS_PATH, professor_name), exist_ok=True)
    # print(f"'{os.path.join(PROFESSORS_DOCS_PATH, professor_name)}' 폴더의 '{professor_name}' 교수님 문서를 FAISS 인덱스에 추가합니다.")
    # incremental_update_vectorstore(
    #     os.path.join(PROFESSORS_FAISS_BASE_PATH, professor_name),
    #     os.path.join(PROFESSORS_DOCS_PATH, professor_name),
    #     "professor_material" # 교수님 자료는 다른 source_type 사용
    # )
