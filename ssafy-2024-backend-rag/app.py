import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
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
            """,
        ),
        ("human", "{input}")
    ]
)

# upstage models
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cards"

# create new index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 5}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

def search_and_show(query, vectorstore, k=3):
    results = vectorstore.similarity_search(query, k=k)

    card_seen = set()
    for doc in results:
        card_name = doc.metadata["card_name"]
        if card_name in card_seen:
            continue  # 같은 카드 이름은 한 번만 출력
        card_seen.add(card_name)

        print(f"카드 이름: {card_name}")
        print(f"카드사: {doc.metadata['card_corp']}")
        print(f"내용 (앞 300자): {doc.metadata['card_full_data'][:300]}...")
        print("---")


# [4] LLM & Prompt 구성 (Upstage LLM)
#llm = ChatUpstage()
llm = ChatUpstage(temperature=0)

chain = prompt | llm | StrOutputParser()

def build_context_from_metadata(results):
    """retriever 결과에서 카드 단위로 card_full_data 기반 context 생성"""
    card_contexts = {}
    for doc in results:
        card_name = doc.metadata["card_name"]
        if card_name not in card_contexts:
            card_contexts[card_name] = doc.metadata["card_full_data"]

    # 카드별 card_full_data를 하나씩 context에 추가
    context = "\n\n".join(
        f"=== {card_name} ===\n{card_data}"
        for card_name, card_data in card_contexts.items()
    )
    return context


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]  # Entire conversation for naive mode


class MessageRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # 🔍 질의 & 응답 실행
    query = req.message
    print("[질문]\n",query)
    results = pinecone_retriever.invoke(query)
    print("[카드 종류]\n",results)
    # metadata 기반 context 생성
    context = build_context_from_metadata(results)

    #print("[context]\n", context, "...")  # 너무 길면 앞 1000자만 출력

    # LLM 호출
    answer = chain.invoke({"context": context, "input": query})

    print("[최종 답변]\n", answer)
    return {"reply": answer}


# @app.post("/assistant")
# async def assistant_endpoint(req: AssistantRequest):
#     assistant = await openai.beta.assistants.retrieve("asst_tc4AhtsAjNJnRtpJmy1gjJOE")
#
#     if req.thread_id:
#         # We have an existing thread, append user message
#         await openai.beta.threads.messages.create(
#             thread_id=req.thread_id, role="user", content=req.message
#         )
#         thread_id = req.thread_id
#     else:
#         # Create a new thread with user message
#         thread = await openai.beta.threads.create(
#             messages=[{"role": "user", "content": req.message}]
#         )
#         thread_id = thread.id
#
#     # Run and wait until complete
#     await openai.beta.threads.runs.create_and_poll(
#         thread_id=thread_id, assistant_id=assistant.id
#     )
#
#     # Now retrieve messages for this thread
#     # messages.list returns an async iterator, so let's gather them into a list
#     all_messages = [
#         m async for m in openai.beta.threads.messages.list(thread_id=thread_id)
#     ]
#     print(all_messages)
#
#     # The assistant's reply should be the last message with role=assistant
#     assistant_reply = all_messages[0].content[0].text.value
#
#     return {"reply": assistant_reply, "thread_id": thread_id}


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
