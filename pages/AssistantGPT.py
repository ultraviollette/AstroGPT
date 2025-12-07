import streamlit as st
import os
import io
from typing import Any, Type, List
import httpx 
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, DuckDuckGoSearchResults, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field

# 1. 전역 상태 및 유틸리티 (st.session_state로 대체)

# Agent가 생성한 최종 텍스트를 저장하기 위한 딕셔너리를 st.session_state로 관리합니다.

@st.cache_resource
def get_llm(openai_api_key):
    """LLM 인스턴스를 캐시하여 재사용합니다."""
    if not openai_api_key:
        return None
    return ChatOpenAI(
        temperature=0.1,
        model="gpt-4o-mini",
        openai_api_key=openai_api_key
    )

# 2. 커스텀 도구 정의

class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
    Use this tool to perform web searches using the DuckDuckGo search engine.
    It takes a query as an argument.
    Example query: "Latest technology news"
    """
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema
    return_direct: bool = False

    def _run(self, query) -> Any:
        # LangChain의 내장 DuckDuckGoSearchResults 사용
        search = DuckDuckGoSearchResults(max_results=3) # 결과를 3개로 제한
        try:
            # HTTPError 처리 로직 추가
            return search.run(query)
        except Exception as e:
            # httpx.HTTPError를 포함한 모든 잠재적인 네트워크/HTTP 오류를 포착합니다.
            if isinstance(e, httpx.HTTPError) or "HTTPError" in str(e):
                return "DuckDuckGo search failed due to a network or server error. Please rely on other tools for this step."
            # 다른 예상치 못한 오류는 다시 발생시킵니다.
            raise e


class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
    Use this tool to perform searches on Wikipedia.
    It takes a query as an argument.
    Example query: "Artificial Intelligence"
    """
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema
    return_direct: bool = False

    def _run(self, query) -> Any:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)

class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")

class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = """
    If you found a potentially useful website link through DuckDuckGo,
    Use this to get the textual content of that link for detailed research.
    """
    args_schema: Type[WebScrapingToolArgsSchema] = WebScrapingToolArgsSchema
    return_direct: bool = False

    def _run(self, url):
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            text = "\n\n".join([doc.page_content for doc in docs])
            
            # 텍스트가 너무 길면 Agent 추론을 위해 일부만 반환
            MAX_CHARACTERS = 4000
            if len(text) > MAX_CHARACTERS:
                return f"Successfully scraped URL: {url}. Extracted content (first {MAX_CHARACTERS} chars): {text[:MAX_CHARACTERS]}..."
                
            return f"Successfully scraped URL: {url}. Extracted content:\n{text}"
        except Exception as e:
            return f"Error scraping URL {url}: {e}"

class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The detailed, final research result text you will save to a file.")

class SaveToTXTTool(BaseTool):
    name = "SaveToTXTTool"
    description = """
    Use this tool to save the *FINAL, COMPLETE* research content as a .txt file.
    This should be the very last step before concluding the research.
    """
    args_schema: Type[SaveToTXTToolArgsSchema] = SaveToTXTToolArgsSchema
    return_direct: bool = True # 이 도구가 호출되면 Agent는 최종 답변을 반환해야 함

    def _run(self, text) -> Any:
        st.session_state.saved_research["content"] = text
        return "Research results successfully prepared and marked for saving. Agent should now proceed with the Final Answer."


# 3. Agent 초기화 및 실행 로직

@st.cache_resource(experimental_allow_widgets=True)
def initialize_research_agent(_llm_instance):
    """시스템 메시지와 도구를 설정하여 Agent를 초기화합니다."""
    if not _llm_instance:
        return None
        
    system_message_content = """
    You are a research expert.
    Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided.
    When you find a relevant website link through DuckDuckGo, you must use the 'WebScrapingTool' to get the content from that website. 
    Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized, detailed, and includes citations with links (URLs) for all sources used.
    Your research should be saved to a .txt file using the 'SaveToTXTTool', and the content should match the detailed findings you provide to the user.
    The information from Wikipedia must be included if relevant.
    You must always call the 'SaveToTXTTool' as the last step before returning the final response.
    """
    
    agent = initialize_agent(
        llm=_llm_instance,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            WebScrapingTool(),
            SaveToTXTTool(),
        ],
        agent_kwargs={"system_message": SystemMessage(content=system_message_content)},
        handle_parsing_errors=True
    )
    return agent

def run_agent_and_update_chat(agent, query):
    """
    Agent를 실행하고 Streamlit 세션 상태를 업데이트합니다.
    """
    
    # Agent 실행 전, 이전 연구 결과를 지우고 새 쿼리를 st.session_state에 기록합니다.**
    st.session_state.saved_research["content"] = None
    st.session_state.saved_research["query"] = query
    
    try:
        # Agent 실행
        result = agent.run(query)
        
        # Agent 결과 저장
        st.session_state.chat_history.append({"role": "assistant", "message": result})
        
    except Exception as e:
        # --- 오류 처리 로직 유지 및 개선 ---
        error_message = f"Agent Run failed: {type(e).__name__}: {e}. Please check your API Key (Quota) or query complexity/network issues."
        
        # 1. 오류 메시지 출력
        st.error(error_message)
        
        # 2. 자세한 트레이스백을 UI에 표시 (디버깅 목적)
        st.exception(e)
        
        # 3. 채팅 기록에 오류 메시지 추가
        st.session_state.chat_history.append({"role": "assistant", "message": f"처리 중 오류가 발생했습니다: {type(e).__name__}: {e}"})


# 4. Streamlit UI

# 4.1. 초기 설정 및 사이드바
st.set_page_config(
    page_title="AssistantGPT - LangChain Research Agent",
    page_icon="🔍",
)

st.title("AssistantGPT")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
# 파일 저장 상태를 위한 st.session_state 초기화
if "saved_research" not in st.session_state:
    st.session_state.saved_research = {"content": None, "query": None}


openai_api_key = None

with st.sidebar:
    st.markdown("## Configuration")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API Key...",
    )
    
    st.markdown("---")
    st.markdown(
        """
        ### About this App
        This application uses a **LangChain Agent (OpenAI Functions)** to perform research.
        It utilizes the following custom tools:
        1.  🌐 **DuckDuckGoSearchTool** (Web search)
        2.  🧠 **WikipediaSearchTool** (Detailed background)
        3.  📄 **WebScrapingTool** (Extract content from URLs)
        4.  💾 **SaveToTXTTool** (Saves final research)
        The Agent combines information from these sources to provide comprehensive research results.
        """
    )

    st.markdown("---")
    st.markdown("[GitHub Repository Link](https://github.com/ultraviollette/AstroGPT)") 
    st.markdown("---")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.saved_research = {"content": None, "query": None} # 저장 상태 초기화
        st.cache_resource.clear() 
        st.rerun()


# 4.2. 메인 로직

# Agent 초기화
llm = get_llm(openai_api_key)
if not llm:
    st.info("⚠️ Please enter your OpenAI API Key in the sidebar to proceed.")
    st.stop()
    
agent = initialize_research_agent(llm)

# 1. 사용자 입력 처리
if prompt := st.chat_input("Ask the agent to research a topic (e.g., 'Research about the XZ backdoor and its impact')..."):
    # 사용자 메시지 기록
    st.session_state.chat_history.append({"role": "user", "message": prompt})
    
    with st.spinner(f"Running Research Agent for: {prompt}"):
        run_agent_and_update_chat(agent, prompt)

# 2. 대화 기록 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# 3. 저장된 콘텐츠 표시 및 다운로드
# st.session_state에서 데이터를 읽어와 다운로드 버튼을 렌더링합니다.
if st.session_state.saved_research["content"]:
    content = st.session_state.saved_research["content"]
    query = st.session_state.saved_research["query"]
    
    st.success("✅ Research complete! The final results have been saved by the Agent.")
    
    # 다운로드 버튼
    download_filename = f"{query.replace(' ', '_')}_research.txt"
    st.download_button(
        label=f"⬇️ Download {download_filename}",
        data=content,
        file_name=download_filename,
        mime="text/plain",
        key='download_research_txt'
    )
    
    # 결과 미리보기
    with st.expander("📝 View Saved Research Content"):
        st.code(content, language='markdown')
    
