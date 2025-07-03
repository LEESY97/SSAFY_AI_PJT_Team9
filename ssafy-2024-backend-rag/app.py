import os
import uuid
import re
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain 관련 모듈 ---
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone, ServerlessSpec

# --- .env 파일에서 환경 변수 로드 ---
load_dotenv()

# --- 1. 전역 객체 및 기록 저장소 설정 ---

# 1-1. llm, embeddings 세팅
llm = ChatUpstage(temperature=0)
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

# 1-2. Pinecone 클라이언트 및 인덱스 초기화
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cards"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=4096, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 5})

# # 1-3. 카드 이름 데이터 로드
# # [데이터 로드]
# with open("all_cards.txt", encoding="utf-8") as f:
#     raw_text = f.read()

# # [카드 단위 분리]
# card_blocks = re.split(r"(?=\[카드 이름\])", raw_text)
# card_blocks = [block.strip() for block in card_blocks if block.strip()]

# # [추가] 전체 카드 이름 목록 미리 생성
# all_card_names = set()
# for block in card_blocks:
#     name_match = re.search(r"\[카드 이름\]\s*(.*)", block)
#     if name_match:
#         all_card_names.add(name_match.group(1).strip())
# # 리스트로 변환하여 사용
# all_card_names = list(all_card_names)
# print(all_card_names)
# print(f"✅ 총 {len(all_card_names)}개의 고유한 카드 이름을 로드했습니다.")

# 1-4. 대화 기록과 제목을 함께 저장하는 저장소
chat_sessions = {}

def get_session_data(session_id: str) -> dict:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": ChatMessageHistory(),
            "title": None,
        }
    return chat_sessions[session_id]


# --- 2. 대화형 RAG 체인 및 '제목 생성 체인' 구성 ---

# 2-1. History-Aware Retriever 구성
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 2-2. 최종 답변 생성을 위한 프롬프트 (기존 프롬프트 사용)
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
            - 답변 맨 마지막에는 항상 '- Cardly Recommending: []'를 만들어줘. '[]'는 너가 추천하는 카드들의 리스트야. 만약 추천하는 카드들이 없을 경우, 그냥 빈 리스트로 둬.


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
            
            - Cardly Recommending: [추천 카드 리스트(추천 카드가 없을 경우 빈 리스트로 출력)]
            
            context 형식:
            [카드 이름] ...
            [카드사] ...
            [혜택 요약] ...

            context는 다음과 같아:
            {context}
"""
qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])

# 2-3. 검색된 문서를 바탕으로 context를 만드는 함수
def format_docs_and_build_context(docs: List[Document]) -> str:
    card_contexts = {}
    for doc in docs:
        card_name = doc.metadata.get("card_name")
        if card_name and card_name not in card_contexts:
            card_data = doc.metadata.get("card_full_data", doc.page_content)
            card_contexts[card_name] = card_data
    return "\n\n".join(f"=== {card_name} ===\n{card_data}" for card_name, card_data in card_contexts.items())

# 2-4. 최종적인 문서 생성 및 답변 체인 
final_answer_chain = qa_prompt | llm | StrOutputParser()

conversational_rag_chain_produces_string = (
    RunnablePassthrough.assign(
        context=history_aware_retriever | RunnableLambda(format_docs_and_build_context)
    )
    | final_answer_chain
)

conversational_rag_chain = RunnablePassthrough.assign(
    answer=conversational_rag_chain_produces_string
)

# 2-5. 대화 내용 요약(제목 생성)을 위한 별도의 체인
title_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 대화 내용을 바탕으로, 전체 대화의 핵심 주제를 5~10단어 이내의 간결한 한글 제목으로 요약해줘. 사용자의 가장 핵심적인 질문이나 요구사항을 담아내. 예를 들어 '스타벅스 할인 카드 추천' 처럼 요약해줘."),
    MessagesPlaceholder("chat_history"),
])
title_chain = title_prompt | llm | StrOutputParser()

# 2-6. 대화 기록 관리 기능과 RAG 체인 최종 결합
chain_with_history = RunnableWithMessageHistory(
    conversational_rag_chain,
    lambda session_id: get_session_data(session_id)["history"],
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# --- 3. FastAPI 애플리케이션 정의 ---

app = FastAPI(title="Card Recommendation Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatEndpointRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# JSON 응답 + 즉시 제목 생성을 처리하는 엔드포인트
@app.post("/chat")
async def chat_endpoint(req: ChatEndpointRequest):
    is_new_conversation = not req.session_id
    session_id = req.session_id or str(uuid.uuid4())
    print(f"[요청] session_id: {session_id}, message: {req.message}")
    
    config = {"configurable": {"session_id": session_id}}

    # 메인 RAG 체인 실행하여 답변 받기
    result = await chain_with_history.ainvoke({"input": req.message}, config=config)
    final_answer = result.get("answer", "오류: 답변을 생성하지 못했습니다.")
    
    # --- title 및 card_list 생성 로직 ---
    card_list = [] # 기본값은 빈 리스트
    # "- Cardly Recommending:[...]" 패턴을 찾기 위한 정규표현식
    match = re.search(r"- Cardly Recommending:\s\[(.*?)\]", final_answer)
    print(match)

    if match:
        # 대괄호 안의 내용(그룹 1)을 추출합니다. ex: "'카드A', '카드B'"
        cards_string = match.group(1)
        if cards_string: # 괄호 안이 비어있지 않은 경우
            # 쉼표로 분리하고, 각 항목의 양쪽 공백과 따옴표를 제거합니다.
            card_list = [
                card.strip().strip("'\"") for card in cards_string.split(',') if card.strip()
            ]

    # [추가 개선] 최종 답변에서 Cardly Recommending 라인 제거
    cleaned_reply = re.sub(r"- Cardly Recommending: .*", "", final_answer).strip()    

    # title 생성 또는 조회 (기존과 동일)
    session_data = get_session_data(session_id)
    title = session_data.get("title")

    print(f"[{session_id}] 대화 시작, 제목을 생성합니다...")
    new_title = await title_chain.ainvoke({"chat_history": session_data["history"].messages})
    session_data["title"] = new_title
    title = new_title
    print(f"[{session_id}] 생성된 제목: {title}")
    print(title)

    print(f"[최종 답변] {final_answer}")

    return {
        "reply": cleaned_reply, 
        "session_id": session_id,
        "title": title,
        "card_list": card_list,
    }

@app.get("/")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)