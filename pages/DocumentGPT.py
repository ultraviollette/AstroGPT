from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

# Callback handler for streaming LLM responses
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# RAG Pipeline
@st.cache_data(show_spinner="Embedding document...")
def embed_file(file):
    # Save uploaded file
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Embedding
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # Initialize vectorstore
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# Streamlit App Configuration
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

st.title("DocumentGPT")

# Sidebar Configuration
with st.sidebar:
    st.markdown("## Configure DocumentGPT")

    # API Key Input
    api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
    )

    # GitHub Link
    st.markdown("[GitHub Repo Link](https://github.com/ultraviollette/AstroGPT/commit/02d2e6976975446db05fc34564a5635924fef94e)")

    # File Uploader
    file = st.file_uploader(
        "Upload your document here",
        type=["pdf", "docx", "txt"],
    )


# Main App Logic
if not api_key:
    st.info("Please enter your OpenAI API key to proceed.")
    st.stop()

if file:
    retriever = embed_file(file)

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=api_key,
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        send_message("You can now ask questions about your document!", "ai", save=False)

    paint_history()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. 
                If you don't know the answer, just say you don't know. 
                Do NOT make anything up.

                Context:
                {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    message = st.chat_input("Ask a question about your document:")

    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.markdown(
        """
        ### Upload a document on the sidebar and ask questions about it!
        """
    )
    st.session_state["messages"] = []