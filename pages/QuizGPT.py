import json
import tempfile
import os # 파일 정리(os.unlink) 및 파일 처리를 위해 필요
from typing import Dict, Any, List # 타입 힌트 유지

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser 
import streamlit as st

# --- 1. Function Calling Schema Definition ---
quiz_function = {
    "name": "create_quiz",
    "description": "Generates a multiple-choice quiz with exactly 10 questions based on context and difficulty.",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The text of the multiple-choice question.",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                        "description": "The potential answer text.",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                        "description": "True if this is the correct answer, False otherwise. Exactly one must be True.",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                            "description": "A list of exactly 4 answers.",
                        },
                    },
                    "required": ["question", "answers"],
                },
                "description": "A list of exactly 10 multiple-choice questions.",
            }
        },
        "required": ["questions"],
    },
}

# --- Streamlit Setup ---

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

# --- Utility Functions ---

def format_docs(docs: List[Any]) -> str:
    """Formats a list of documents into a single string for the prompt."""
    return "\n\n".join(document.page_content for document in docs)

def get_llm(api_key: str):
    """Initializes the ChatOpenAI model, binding it to the quiz generation function."""
    if not api_key:
        return None
        
    # Function Calling을 사용하고, JSON 출력을 위해 스트리밍은 비활성화
    llm = ChatOpenAI(
        temperature=0.3, 
        model="gpt-4-turbo",
        openai_api_key=api_key,
        streaming=False, 
    ).bind(
        function_call={"name": "create_quiz"},
        functions=[quiz_function],
    )
    return llm

# --- Chains and Prompts ---

# 1. Question Generation Prompt (Function Calling에 맞춰 간결하게 수정)
questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
    Your task is to create a quiz based *only* on the provided context.
    
    You MUST use the `create_quiz` function to output the result.
    
    **Difficulty Level:** {difficulty}
    
    Create exactly 10 multiple-choice questions. Ensure each question has exactly 4 answers, with one being factually correct based on the provided context.
    
    **IMPORTANT:** Generate the quiz, including all questions and answers, in the same language as the provided Context.
    
    Context: {context}
""",
        )
    ]
)

# --- Data Loading Functions ---

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    """Loads and splits the uploaded file content using a temporary file (required by UnstructuredFileLoader)."""
    # tempfile을 사용하여 임시 파일 생성 및 처리
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        # UnstructuredFileLoader를 사용하여 파일 로드 및 분할
        loader = UnstructuredFileLoader(tmp_file_path) 
        docs = loader.load_and_split(text_splitter=splitter)
        return docs
    finally:
        # 임시 파일 삭제
        os.unlink(tmp_file_path) 

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term: str) -> List[Any]:
    """Retrieves relevant documents from Wikipedia."""
    retriever = WikipediaRetriever(top_k_results=5) 
    docs = retriever.get_relevant_documents(term)
    return docs

def run_quiz_chain(docs: List[Any], difficulty: str, api_key: str):
    """Runs the chain using Function Calling to generate the structured quiz."""
    llm = get_llm(api_key)
    if not llm:
        return None

    # Function Calling 바인딩된 LLM과 질문 프롬프트 연결
    chain = (
        {"context": format_docs, "difficulty": lambda x: difficulty} 
        | questions_prompt 
        | llm
    )
    
    try:
        # 체인 실행
        response = chain.invoke(docs)
        
        # Function Calling 응답 파싱
        if response.additional_kwargs.get("function_call"):
            function_args_str = response.additional_kwargs["function_call"]["arguments"]
            # JSON 문자열을 딕셔너리로 로드
            quiz_data = json.loads(function_args_str)
            return quiz_data
        
        # Function Call이 없는 경우 (오류 상황)
        raise ValueError("Model failed to call the 'create_quiz' function.")
    
    except Exception as e:
        st.error(f"Error generating or parsing quiz: {e}. Ensure your API key is valid and the model returned valid JSON.")
        print(f"Quiz Generation/Parsing Error: {e}") 
        return None

# --- Session State Initialization ---

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "score" not in st.session_state:
    st.session_state.score = 0
if "max_score" not in st.session_state:
    st.session_state.max_score = 0
if "source_name" not in st.session_state:
    st.session_state.source_name = "Untitled Source"
# 만점 시 축하 효과를 위한 플래그 추가
if "show_balloons" not in st.session_state:
    st.session_state.show_balloons = False
# 문서 데이터를 st.session_state에 저장할 새로운 변수 초기화
if "document_data" not in st.session_state:
    st.session_state.document_data = None
# 파일 업로더의 이전 상태를 추적하는 변수
if "file_uploader_value" not in st.session_state:
    st.session_state.file_uploader_value = None
# Wikipedia의 이전 검색어를 추적하는 변수
if "last_wiki_topic" not in st.session_state:
    st.session_state.last_wiki_topic = ""


# --- Sidebar Configuration ---

with st.sidebar:
    st.header("1. Settings")
    
    # API Key Input
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Enter your API key to generate the quiz.",
        key="api_key_input"
    )
    
    # Difficulty Selector
    st.session_state.difficulty = st.selectbox(
        "Select Quiz Difficulty",
        ("Easy", "Medium", "Hard"),
        index=0,
        key="difficulty_select"
    )
    
    st.divider()
    st.header("2. Content Source")

    
    uploaded_file = None
    
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
        key="source_choice"
    )
    
    if choice == "File":
        uploaded_file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file",
            type=["pdf", "txt", "docx"],
            key="file_uploader"
        )
        
        # 파일이 업로드되었고, 이전 파일과 다르다면 데이터를 로드합니다.
        if uploaded_file is not None and uploaded_file != st.session_state.file_uploader_value:
            st.session_state.document_data = split_file(uploaded_file)
            st.session_state.source_name = uploaded_file.name
            st.session_state.quiz_data = None # 새 파일 로드 시 퀴즈 초기화
            st.session_state.submitted = False
            st.session_state.show_balloons = False # 만점 플래그 초기화
            st.session_state.file_uploader_value = uploaded_file # 새 파일 상태 추적

        # 사용자가 업로더를 지웠다면 문서 데이터도 초기화
        elif uploaded_file is None and st.session_state.document_data is not None:
             st.session_state.document_data = None
             st.session_state.quiz_data = None
             st.session_state.submitted = False
             st.session_state.show_balloons = False # 만점 플래그 초기화
             st.session_state.file_uploader_value = None
            
    else: # Wikipedia Article
        topic = st.text_input("Search Wikipedia...", key="wiki_topic_input")
        
        # --- 버그 수정: Topic이 변경된 경우에만 검색을 실행하고 상태를 초기화합니다. ---
        if topic and topic != st.session_state.last_wiki_topic:
            # 토픽이 비어있지 않고, 이전 검색어와 다르다면 검색을 실행하고 데이터를 로드합니다.
            st.session_state.document_data = wiki_search(topic)
            st.session_state.source_name = topic
            st.session_state.last_wiki_topic = topic # 새로운 검색어 저장
            st.session_state.quiz_data = None # 새 검색 시 퀴즈 초기화
            st.session_state.submitted = False
            st.session_state.show_balloons = False # 만점 플래그 초기화
            
        # 검색어가 없고 document_data가 있다면 초기화
        elif not topic and st.session_state.document_data is not None:
             st.session_state.document_data = None
             st.session_state.quiz_data = None
             st.session_state.submitted = False
             st.session_state.show_balloons = False # 만점 플래그 초기화
             st.session_state.last_wiki_topic = "" # 검색어 초기화
        # ------------------------------------------------------------------------
            
    # --- Quiz Generation Button Logic ---
    
    # st.session_state.document_data가 있고 퀴즈 데이터가 없을 때 버튼 표시
    if st.session_state.quiz_data is None and st.session_state.document_data:
        if st.button("Generate Quiz", key="generate_quiz_btn"):
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API Key first.")
            else:
                with st.spinner(f"Generating a {st.session_state.difficulty} quiz from '{st.session_state.source_name}'..."):
                    # st.session_state.document_data를 함수에 전달
                    quiz_result = run_quiz_chain(st.session_state.document_data, st.session_state.difficulty, st.session_state.openai_api_key)
                    
                    if quiz_result and "questions" in quiz_result:
                        st.session_state.quiz_data = quiz_result
                        st.session_state.submitted = False 
                        st.session_state.score = 0
                        st.session_state.max_score = len(st.session_state.quiz_data["questions"]) 
                        st.session_state.show_balloons = False # 만점 플래그 초기화
                    else:
                         # 오류는 run_quiz_chain에서 처리되었으므로 추가 메시지는 생략
                         st.session_state.quiz_data = None
    
    # GitHub Link
    st.markdown("---")
    st.markdown("[GitHub Repo Link](https://github.com/ultraviollette/fullstack-gpt)") 

# --- Main Quiz Interface Logic ---

def check_answers_and_update_state(quiz_data: Dict[str, Any], user_answers: Dict[str, str]):
    """Checks answers after submission and updates session state."""
    correct_count = 0
    max_count = len(quiz_data["questions"])
    
    for i, question in enumerate(quiz_data["questions"]):
        user_selection = user_answers.get(f"q_{i}", None)
        
        # Find the correct answer text by checking the 'correct' flag in the answers list
        correct_answer_text = next(
            (ans["answer"] for ans in question["answers"] if ans["correct"]), 
            None
        )

        # Check if user selection matches the correct answer
        if user_selection and user_selection == correct_answer_text:
            correct_count += 1
            
    st.session_state.score = correct_count
    st.session_state.max_score = max_count
    st.session_state.submitted = True
    
    # 만점일 경우, 다음 실행 시 축하 효과를 트리거하도록 플래그 설정
    if correct_count == max_count:
        st.session_state.show_balloons = True
    
    # 참고: st.balloons() 및 성공/경고 메시지 출력은 이 함수 밖의 메인 로직에서 처리됨

def retake_quiz():
    """Resets the submission state to allow the user to retake the test."""
    # Retake 시 제출 상태와 만점 플래그 초기화하고 퀴즈 데이터는 유지
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.show_balloons = False

# Initial welcome message
if st.session_state.document_data is None and st.session_state.quiz_data is None:
    st.markdown(
        """
    Welcome to **QuizGPT (Function Calling Edition)**.
                
    I will make a customized quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    **To start:**
    1. Enter your OpenAI API Key in the sidebar.
    2. Choose a difficulty level.
    3. Upload a file or search for a Wikipedia topic.
    4. Click 'Generate Quiz' to begin your test!
    """
    )
elif st.session_state.quiz_data is None:
    # Content is loaded (document_data exists), but quiz hasn't been generated yet
    st.info(f"Content loaded from '{st.session_state.source_name}'. Click 'Generate Quiz' in the sidebar to start the test.")
    
else:
    # Quiz is generated, display the form
    
    # 제출 후에도 st.radio의 선택 값을 유지하기 위해 form 밖에서 딕셔너리를 준비합니다.
    user_answers = {} 

    with st.form("questions_form"):
        st.subheader(f"Quiz on **{st.session_state.source_name}** (Difficulty: {st.session_state.difficulty})")
        
        for i, question in enumerate(st.session_state.quiz_data["questions"]):
            st.markdown(f"**{i+1}.** {question['question']}")
            
            correct_answer_text = next(
                (ans["answer"] for ans in question["answers"] if ans["correct"]), 
                None
            )
            
            radio_key = f"q_{i}"
            
            if not st.session_state.submitted:
                # 제출 전: st.radio를 사용하여 사용자 입력 허용
                value = st.radio(
                    "Select an answer:",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=radio_key, # 이 키에 선택 값이 저장됨
                    label_visibility="collapsed"
                )
                user_answers[radio_key] = value
            else:
                # 제출 후: 커스텀 HTML로 피드백 표시
                user_selection = st.session_state.get(radio_key) 
                # 정답 확인 함수를 위해 user_answers에 값을 다시 할당
                user_answers[radio_key] = user_selection 

                for answer_option in question["answers"]:
                    answer_text = answer_option["answer"]
                    
                    if answer_text == correct_answer_text:
                        # 정답 (초록색 하이라이트)
                        icon = "✅"
                        style = "background-color: #e6ffe6; border-left: 5px solid green; padding: 10px; margin-bottom: 5px; border-radius: 4px; color: #1e7e34;"
                    elif answer_text == user_selection:
                        # 사용자가 선택한 오답 (빨간색 하이라이트)
                        icon = "❌"
                        style = "background-color: #ffe6e6; border-left: 5px solid red; padding: 10px; margin-bottom: 5px; border-radius: 4px; color: #dc3545;"
                    else:
                        # 나머지 오답 옵션 (기본 스타일)
                        icon = "•"
                        style = "padding: 10px; margin-bottom: 5px; border-radius: 4px; border: 1px solid #f0f0f0; color: #333333;"

                    st.markdown(
                        f'<div style="{style}">{icon} {answer_text}</div>', 
                        unsafe_allow_html=True
                    )
                
                # 문제별 최종 피드백 표시
                if user_selection == correct_answer_text:
                    st.success("✅ Correct!")
                elif user_selection is not None:
                    st.error(f"❌ Incorrect. The correct option is marked above.")
                else:
                    st.info("You skipped this question. The correct option is marked above.")

            st.markdown("---")
        
        # Submit Button
        submit_button = st.form_submit_button("Submit Answers", 
                                              disabled=st.session_state.submitted)
        
        if submit_button and not st.session_state.submitted:
            # 폼 제출 시, st.session_state에 저장된 라디오 버튼 값을 가져옵니다.
            answers_from_state = {f"q_{i}": st.session_state.get(f"q_{i}") 
                                    for i in range(len(st.session_state.quiz_data["questions"]))}
            check_answers_and_update_state(st.session_state.quiz_data, answers_from_state)
            
            # 상태 업데이트 후 즉시 앱을 재실행하여 결과를 화면에 바로 표시
            st.rerun() 


    # Score and Retake Buttons (outside the form)
    if st.session_state.submitted:
        st.markdown("---")
        
        # 만점 축하 효과를 여기서 트리거합니다.
        if st.session_state.show_balloons:
            st.balloons() # 만점 시 st.balloons 사용
            st.session_state.show_balloons = False # 효과는 한 번만 표시되도록 플래그 재설정
            st.success(f"🥳 Congratulations! You got all {st.session_state.max_score} questions correct! Well done!")
        
        # 만점이 아닐 경우 일반적인 경고 메시지 표시
        elif st.session_state.score < st.session_state.max_score:
            st.warning(f"Quiz submitted. You scored {st.session_state.score} out of {st.session_state.max_score}. Check the questions below for detailed feedback.")
        
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.metric(
                label="Your Score", 
                value=f"{st.session_state.score} / {st.session_state.max_score}",
            )
            
        with col2:
            if st.session_state.max_score > 0:
                accuracy = st.session_state.score / st.session_state.max_score * 100
                st.write(f"Accuracy: **{accuracy:.1f}%**")
        
        # Retake 버튼
        st.button("Retake Test", on_click=retake_quiz)