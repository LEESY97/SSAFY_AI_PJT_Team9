import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain 관련 모듈 ---
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone, ServerlessSpec

# --- .env 파일에서 환경 변수 로드 ---
load_dotenv()

# --- 1. 전역 객체 및 기록 저장소 설정 ---

llm = ChatUpstage(streaming=True)
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cards"

if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=4096, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 5})

chat_histories = {}

# [수정 확인] 함수 인자가 session_id 인지 확인
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """session_id를 기반으로 대화 기록을 가져오거나 새로 생성합니다."""
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]


# --- 2. 대화형 RAG 체인 구성 ---

contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """
            너는 신용카드 추천 전문가야.
            아래 규칙을 반드시 지켜:
            - 신용카드 추천과 관련되지 않은 내용은 절대 언급하지 마.
            - context에 주어진 카드 정보만 기반으로 답변해.
            - context에 없는 카드 정보는 사용하지 마.
            - '[카드 이름]'을 포함하지 않는 context는 사용하지마.
            - 답변에 불필요한 설명이나 사견을 넣지 마.
            - 카드 이름, 카드사, 추천 이유, 장점, 단점을 반드시 모두 작성해.
            - 추천 이유나 장점은 디테일하게 설명해줘.
            - 카드 정보를 3개 이하로 제공해.
            - 답변을 한 후 추가로 원하는 조건을 다시 물어봐.
            - 적합한 카드가 없으면 카드 정보를 출력하지 않고, 카드 추천을 위한 질문을 해


            출력 형식:
            1. 카드 이름 : [카드 이름]
               카드사 : [카드사]
               추천 이유 : [추천 핵심 이유]
               장점 : [구체적 혜택]
               단점 : [제한사항]

            2. 카드 이름 : ...
               카드사 : ...
               추천 이유 : ...
               장점 : ...
               단점 : ...

            3. 카드 이름 : ...
               카드사 : ...
               추천 이유 : ...
               장점 : ...
               단점 : ...

            context 형식:
            [카드 이름] ...
            [카드사] ...
            [혜택 요약] ...

            context는 다음과 같아:
            {context}
"""
qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])

def format_docs_and_build_context(docs: List[Document]) -> str:
    card_contexts = {}
    for doc in docs:
        card_name = doc.metadata.get("card_name")
        if card_name and card_name not in card_contexts:
            card_data = doc.metadata.get("card_full_data", doc.page_content)
            card_contexts[card_name] = card_data
    return "\n\n".join(f"=== {card_name} ===\n{card_data}" for card_name, card_data in card_contexts.items())

Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
conversational_rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

chain_with_history = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# --- 3. FastAPI 애플리케이션 정의 ---

app = FastAPI(title="Card Recommendation Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# [수정 확인] 요청 Body 모델
class ChatEndpointRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# [수정] JSON 응답을 반환하는 FastAPI 엔드포인트
@app.post("/chat")
async def chat_endpoint(req: ChatEndpointRequest):
    # session_id가 없으면 새로 생성 (대화 기록 관리를 위해 유지)
    session_id = req.session_id or str(uuid.uuid4())
    print(f"[요청] session_id: {session_id}, message: {req.message}")
    
    # LangChain에 전달할 config 설정
    config = {"configurable": {"session_id": session_id}}

    # .ainvoke()를 사용하여 전체 답변이 생성될 때까지 기다렸다가 결과를 받음
    result = await chain_with_history.ainvoke(
        {"input": req.message}, 
        config=config
    )

    # 체인 실행 결과에서 'answer' 키의 값을 추출
    final_answer = result.get("answer", "오류: 답변을 생성하지 못했습니다.")
    
    print(f"[최종 답변] {final_answer}")

    # 최종 답변과 session_id를 JSON 형식으로 반환
    return {"reply": final_answer, "session_id": session_id}


@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)