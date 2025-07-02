import os
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader
from langchain_upstage import UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# data preprocessing
# [1] 데이터 로드
with open("all_cards.txt", encoding="utf-8") as f:
    raw_text = f.read()

# [2] 카드 단위 분리
card_blocks = re.split(r"(?=\[카드 이름\])", raw_text)
card_blocks = [block.strip() for block in card_blocks if block.strip()]

# [3] tokenizer 준비 (token estimate용)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# [4] 카드 단위 sub-chunk + Document 생성
def create_card_documents(card_block):
    # 카드 이름, 카드사 추출
    name_match = re.search(r"\[카드 이름\]\s*(.*)", card_block)
    corp_match = re.search(r"\[카드사\]\s*(.*)", card_block)
    card_name = name_match.group(1).strip() if name_match else "Unknown"
    card_corp = corp_match.group(1).strip() if corp_match else "Unknown"

    # 1차 의미 단위: Option 기준 split
    sub_chunks = re.split(r"(?=Option \d+\.)", card_block)
    if len(sub_chunks) == 1:
        # Option이 없으면 문단 단위
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1500,
            chunk_overlap=100
        )
        sub_chunks = splitter.split_text(card_block)

    documents = []

    for sub in sub_chunks:
        sub = sub.strip()
        if not sub:
            continue
        # token 수 체크
        token_len = len(tokenizer.encode(sub))
        if token_len <= 4000:
            documents.append(
                Document(
                    page_content=sub,
                    metadata={
                        "card_name": card_name,
                        "card_corp": card_corp,
                        "card_full_data": card_block
                    }
                )
            )
        else:
            # token limit 초과 시 더 잘게 나누기
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=1000,  # 더 작게 나눔
                chunk_overlap=100
            )
            smaller_chunks = splitter.split_text(sub)
            for small in smaller_chunks:
                small = small.strip()
                if not small:
                    continue
                small_token_len = len(tokenizer.encode(small))
                if small_token_len > 4000:
                    print(f"⚠️ 매우 큰 조각 감지, token 수: {small_token_len}, 카드: {card_name}")
                    continue  # 이 경우도 추가로 나누거나 경고
                documents.append(
                    Document(
                        page_content=small,
                        metadata={
                            "card_name": card_name,
                            "card_corp": card_corp,
                            "card_full_data": card_block
                        }
                    )
                )

    return documents

# [5] Document 리스트 생성
all_documents = []
for block in card_blocks:
    all_documents.extend(create_card_documents(block))

# upstage models
embedding_upstage = UpstageEmbeddings(model="solar-embedding-1-large")

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

print("start")


# Embed the splits

PineconeVectorStore.from_documents(
    all_documents, embedding_upstage, index_name=index_name
)
print("end")
