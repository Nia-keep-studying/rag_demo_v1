import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_loader import load_documents, split_documents
from rag_search import build_context, retrieve_results, merge_results
from rerank_demo import load_rerank_model,rerank_chunks

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


use_rerank = True
vector_top_k = 5
vector_min_score = 0.4
rerank_top_k = 3


question = input("请输入问题：")
documents = load_documents()
chunks = split_documents(documents)
embedding_model = load_embedding_model()
chunk_vectors = build_chunk_vectors(chunks, embedding_model)




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
    top_k=vector_top_k,
    min_score=vector_min_score,
)
for result in vector_results:
    chunk = result["chunk"]

    print(
        "向量找回：",
        chunk["source"],
        chunk["section"],
        result["score"],
    )

keyword_results = retrieve_results(question, chunks)
for chunk in keyword_results:
    print(
        "关键词找回：",
        chunk["source"],
        chunk["section"],
    )
all_results = merge_results(keyword_results, vector_results)

if not all_results:
    answer = "没有检索到足够相关的资料，暂时无法确认。"
else:
    final_chunks = all_results

    if use_rerank:
        reranker = load_rerank_model()
        rerank_results = rerank_chunks(question=question,chunks=all_results,reranker=reranker,top_k=rerank_top_k)

        final_chunks = []
        for result in rerank_results:
            final_chunks.append(result["chunk"])

    print("最终交给模型的片段")
    for chunk in final_chunks:
        print(chunk["source"],chunk["section"])
    messages = build_messages(question, final_chunks)
    answer = answer_question(messages)

print("最终回答：", answer)