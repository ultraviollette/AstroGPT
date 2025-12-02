import os
from typing import List, Any, Dict
import streamlit as st

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# --- Cloudflare Product Sitemaps ---
CLOUDFLARE_SITEMAPS = [
    "https://developers.cloudflare.com/ai-gateway/sitemap.xml",
    "https://developers.cloudflare.com/vectorize/sitemap.xml",
    "https://developers.cloudflare.com/workers-ai/sitemap.xml",
]

# --- LLM and Embeddings Initialization (Dependent on API Key) ---
# api_key 에러가 자꾸 나서 함수로 분리하여 호출 시점에 환경변수에서 읽도록 수정했습니다

def get_llm():
    """Initializes and returns the ChatOpenAI instance, relying on the OPENAI_API_KEY environment variable."""
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4-turbo"
    )

def get_embeddings():
    """Initializes and returns the OpenAIEmbeddings instance, relying on the OPENAI_API_KEY environment variable."""
    return OpenAIEmbeddings()


# --- Prompts ---

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

# --- Chain Logic Functions ---

def get_answers(inputs):
    """Generates answers for each retrieved document and includes source metadata."""
    docs: List[Document] = inputs["docs"]
    question: str = inputs["question"]
    llm = get_llm()
    answers_chain = answers_prompt | llm

    # 각 문서 조각에 대해 LLM을 호출하여 답변과 점수를 생성
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
    }

def choose_answer(inputs):
    """Selects the best final answer from the generated answers based on score and recency."""
    answers: List[Dict[str, Any]] = inputs["answers"]
    question: str = inputs["question"]
    llm = get_llm()
    choose_chain = choose_prompt | llm
    
    # 여러 답변을 하나의 문자열로 압축 (Choose Prompt에 맞게 포맷팅)
    condensed = "\n\n".join(
            f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}"
            for answer in answers
    )
    
    # 최종 답변 선택 LLM 호출
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    """Removes irrelevant elements (header, footer) from the page content."""
    # Cloudflare 문서에 맞춰 불필요한 요소 제거
    main_content = soup.find("main") or soup
    
    # 불필요한 네비게이션 및 헤더 제거 (SitemapLoader가 가져온 전체 HTML에서)
    for tag in main_content.find_all(['header', 'footer', 'nav', 'aside', 'script', 'style']):
        tag.decompose()
        
    return (
        str(main_content.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .strip()
    )

@st.cache_resource(show_spinner="Loading Cloudflare documentation (may take a moment)...")
def load_website(api_key: str):
    """Loads, splits, and embeds all Cloudflare documentation sitemaps."""
    if not api_key:
        return None

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    all_docs = []
    
    # URL 필터링을 위한 패턴 설정
    url_filters = [
        r"https://developers\.cloudflare\.com/ai-gateway/",
        r"https://developers\.cloudflare\.com/vectorize/",
        r"https://developers\.cloudflare\.com/workers-ai/",
    ]

    # 3개의 sitemap에서 문서를 순차적으로 로드
    for url in CLOUDFLARE_SITEMAPS:
        st.write(f"Loading documentation from: {url}")
        try:
            loader = SitemapLoader(
                url, 
                parsing_function=parse_page,
                filter_urls=url_filters # URL 필터링 적용
            )

            loader.requests_per_second = 15 
            docs = loader.load_and_split(text_splitter=splitter)
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"Failed to load documents from {url}: {e}")
            
    if not all_docs:
        st.error("Could not load any documentation. Please check the URLs or try again later.")
        return None

    st.write(f"Total {len(all_docs)} chunks loaded and ready for embedding.")
    
    # 임베딩 생성 (인수를 전달하지 않음)
    embeddings = get_embeddings()
    
    # 🌟 토큰 한도 초과 오류 방지를 위해 문서 배치를 나누어 임베딩합니다.
    # 안전한 배치 크기를 250개 문서로 설정 (300k 토큰 제한 이하를 목표)
    BATCH_SIZE = 250 
    vector_store = None
    
    # Streamlit 진행 표시줄 설정
    embedding_status = st.empty()
    embedding_progress = st.progress(0)
    
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        current_progress = (i + len(batch)) / len(all_docs)
        
        embedding_status.info(f"Embedding batch {i // BATCH_SIZE + 1} of {len(all_docs) // BATCH_SIZE + 1}...")
        
        try:
            if vector_store is None:
                # 첫 번째 배치: FAISS 저장소 초기화
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                # 후속 배치: 기존 저장소에 문서 추가
                vector_store.add_documents(batch)
            
            # 진행 상태 업데이트
            embedding_progress.progress(current_progress)

        except Exception as e:
            st.error(f"Error during embedding batch {i // BATCH_SIZE + 1}: {e}")
            # 오류가 발생하면 중단하고 None 반환
            return None

    # 임베딩 완료 메시지
    embedding_status.success("Document embedding complete!")
    
    return vector_store.as_retriever()

# --- Streamlit UI and Execution ---

st.set_page_config(
    page_title="Cloudflare SiteGPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # SiteGPT for Cloudflare's documentation
            
    Ask questions about the documentation for Cloudflare's **AI Gateway**, **Vectorize**, and **Workers AI**.
            
    The system will retrieve relevant context from the documentation and use it to provide grounded answers.
"""
)

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to run the application.",
        key="api_key_input"
    )
    st.markdown("---")
    st.markdown("[GitHub Repository Link](https://github.com/ultraviollette/AstroGPT)") 
    st.markdown("---")


# --- Main Application Logic ---

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to start.")
else:
    # API 키를 전역 환경 변수로 설정하여 모든 LangChain 컴포넌트가 사용하도록 보장
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 1. 문서 로드 및 리트리버 초기화 (API 키 필요)
    retriever = load_website(api_key)
    
    if retriever:
        # 2. 질문 입력
        query = st.text_input("Ask a question about the Cloudflare documentation", key="query_input")
        
        if query:
            # 3. RAG 체인 구성 및 실행
            
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers) # get_answers는 이제 환경 변수에서 키를 가져옵니다.
                | RunnableLambda(choose_answer) # choose_answer도 마찬가지입니다.
            )
            
            with st.spinner("Searching and generating answer..."):
                try:
                    # 최종 결과 호출
                    result = chain.invoke(query)
                    
                    # Markdown 결과 표시. $ 문자가 LaTeX 수식으로 해석되지 않도록 이스케이프 처리
                    st.markdown(result.content.replace("$", "\$"))
                    
                except Exception as e:
                    st.error(f"An error occurred during chain execution: {e}")
                    # API 키가 잘못된 경우, 명확하게 안내합니다.
                    st.info("Please check if your API key is correct and valid for the specified model.")
    else:
        st.error("Could not initialize the documentation retriever. Please ensure the API key is valid.")