import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_loader import load_documents, split_documents
from rag_search import search_by_keyword, build_context
from embedding_demo import (
    load_embedding_model,
    build_chunk_vectors,
    search_by_vector,
)

load_dotenv()

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"),
    base_url = "https://api.deepseek.com"
)

question = input("请输入问题：")
documents = load_documents()
chunks = split_documents(documents)
embedding_model = load_embedding_model()
chunk_vectors = build_chunk_vectors(chunks, embedding_model)

def retrieve_results(question,chunks):
    keywords = ["发货","退货","地址","退款"]
    all_results = []
    for keyword in keywords:
        if keyword in question:
            result = search_by_keyword(chunks=chunks,keyword=keyword)
            all_results.extend(result)

    unique_results = []
    seen = set()

    for result in all_results:
        result_id = (result["source"], result["chunk_id"])

        if result_id not in seen:
            unique_results.append(result)
            seen.add(result_id)
    return unique_results


def build_messages(question,all_results):
    messages = [{"role":"system","content":"只能根据参考资料回答；资料不能支持答案时，明确说无法确认；回答末尾写出来源文件和章节。"}]

    if not all_results:
        messages.append({"role":"user","content":f"用户问题：{question};参考资料:没有找到相关资料"})
        return messages
    final_context = build_context(all_results)
    messages.append({"role":"user","content":f"用户问题：{question};参考资料:{final_context}"})
    return messages

def answer_question(messages):
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = messages,
    )
    return response.choices[0].message.content


query_vector = embedding_model.encode([question])

vector_results = search_by_vector(
    query_vector=query_vector,
    chunks=chunks,
    chunk_vectors=chunk_vectors,
    top_k=3,
    min_score=0.6,
)

all_results = []

for result in vector_results:
    chunk = result["chunk"]
    all_results.append(chunk)

    print(
        "向量找回：",
        chunk["source"],
        chunk["section"],
        result["score"],
    )

if not all_results:
    answer = "没有检索到足够相关的资料，暂时无法确认。"
else:
    messages = build_messages(question, all_results)
    answer = answer_question(messages)

print("最终回答：", answer)
