import os
import re
import json
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageDocumentParseLoader, UpstageEmbeddings
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from langchain_core.documents import Document

load_dotenv()

# [1] 데이터 로드
with open("all_cards.txt", encoding="utf-8") as f:
    raw_text = f.read()

# [2] 카드 단위 분리
card_blocks = re.split(r"(?=\[카드 이름\])", raw_text)
card_blocks = [block.strip() for block in card_blocks if block.strip()]

# [2-1] 메타데이터 크기 사전 체크
for block in card_blocks:
    name_match = re.search(r"\[카드 이름\]\s*(.*)", block)
    corp_match = re.search(r"\[카드사\]\s*(.*)", block)
    card_name = name_match.group(1).strip() if name_match else "Unknown"
    card_corp = corp_match.group(1).strip() if corp_match else "Unknown"

    metadata = {
        "card_name": card_name,
        "card_corp": card_corp,
        "card_full_data": block
    }
    metadata_size = len(json.dumps(metadata, ensure_ascii=False).encode('utf-8'))
    if metadata_size >= 30 * 1024:
        print(f"⚠️ 매우 큰 메타데이터 감지 (JSON 기준): {metadata_size / 1024:.2f}KB, 카드 이름: {card_name}")

# [3] tokenizer 준비
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# [4] 토큰 기준 강제 분할 함수
def hard_split_by_tokens(text, max_tokens):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# [5] 카드 단위 Document 생성
def create_card_documents(card_block):
    name_match = re.search(r"\[카드 이름\]\s*(.*)", card_block)
    corp_match = re.search(r"\[카드사\]\s*(.*)", card_block)
    card_name = name_match.group(1).strip() if name_match else "Unknown"
    card_corp = corp_match.group(1).strip() if corp_match else "Unknown"

    sub_chunks = re.split(r"(?=Option \d+\.)", card_block)
    if len(sub_chunks) == 1:
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1500,
            chunk_overlap=0
        )
        sub_chunks = splitter.split_text(card_block)

    documents = []
    for sub in sub_chunks:
        sub = sub.strip()
        if not sub:
            continue

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
            print(f"⚠️ 매우 큰 조각 감지, token 수: {token_len}, 카드: {card_name}")

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " ", ""],
                chunk_size=3000,
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
                    hard_chunks = hard_split_by_tokens(small, 3000)
                    for idx, hard in enumerate(hard_chunks):
                        hard = hard.strip()
                        if not hard:
                            continue
                        hard_token_len = len(tokenizer.encode(hard))
                        if hard_token_len > 4000:
                            print(f"⚠️ [오류] 강제 분할 후에도 4000 초과, token 수: {hard_token_len}, 카드: {card_name}")
                        else:
                            print(f"✅ 강제 분할 조각 {idx+1}/{len(hard_chunks)}, token 수: {hard_token_len}, 카드: {card_name}")
                            documents.append(
                                Document(
                                    page_content=hard,
                                    metadata={
                                        "card_name": card_name,
                                        "card_corp": card_corp,
                                        "card_full_data": card_block
                                    }
                                )
                            )
                else:
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

# [6] Document 리스트 생성
all_documents = []
for block in card_blocks:
    all_documents.extend(create_card_documents(block))

# [7] Upstage 임베딩 및 Pinecone 저장
embedding_upstage = UpstageEmbeddings(model="solar-embedding-1-large")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cards"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start")

database = PineconeVectorStore.from_documents(
    documents=[], 
    embedding=embedding_upstage, 
    index_name=index_name,
)

# Upload documents in batches
batch_size = 100
print(f'total batch:{len(all_documents)}, batch size:{batch_size}')
for i in range(0, len(all_documents), batch_size):
    print(f'index: {i}, batch size: {batch_size}')
    batch = all_documents[i:i + batch_size]
    database.add_documents(batch)  # Add documents to the existing database

print("end")