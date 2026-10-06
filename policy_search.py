from embedding_demo import (
    load_embedding_model,
    build_chunk_vectors,
    search_by_vector,
)
from rag_loader import load_documents, split_documents
from rag_search import retrieve_results, merge_results
from rerank_demo import load_rerank_model, rerank_chunks

def retrieve_policy(question,chunks,model,chunk_vectors,reranker):
    query_vector = model.encode([question])
    retrieve_vector = search_by_vector(query_vector=query_vector,chunks=chunks,chunk_vectors=chunk_vectors,top_k=5,min_score=0.4)
    keyword_results = retrieve_results(question=question,chunks=chunks)

    final_results = merge_results(keyword_results=keyword_results,vector_results=retrieve_vector)

    if not final_results:
        return {"chunks":[]}

    reranked_results = rerank_chunks(question=question,chunks=final_results,reranker=reranker,top_k=3)

    selected_chunks = []
    for result in reranked_results:
        selected_chunks.append(result["chunk"])
    return {"chunks":selected_chunks}



if __name__ == "__main__":
    question = input("请输入要查询的问题：")
    reranker = load_rerank_model()
    documents = load_documents()
    chunks = split_documents(documents)
    embedding_model = load_embedding_model()
    chunk_vectors = build_chunk_vectors(chunks,embedding_model)

    results = retrieve_policy(question=question,chunks=chunks,model=embedding_model,chunk_vectors=chunk_vectors,reranker=reranker)
    if not results["chunks"]:
        print("没有检索到候选资料")
    else:
        for chunk in results["chunks"]:
            print(chunk["source"], chunk["section"], chunk["content"])
