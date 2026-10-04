import os
from pathlib import Path
import numpy as np
from rag_loader import load_documents,split_documents

os.environ["HF_HOME"] = str(Path(__file__).parent / ".model_cache")
from model2vec import StaticModel

def load_embedding_model():
    model = StaticModel.from_pretrained(
        "cnmoro/multilingual-e5-small-distilled-16m",
        normalize = True,
    )
    return model

def build_chunk_vectors(chunks, model):
    chunk_texts = []
    for chunk in chunks:
        text = f"章节:{chunk['section']}\n正文:{chunk['content']}"
        chunk_texts.append(text)

    chunk_vectors = model.encode(chunk_texts)
    return chunk_vectors

def get_score(result):
    return result["score"]

def search_by_vector(query_vector,chunks,chunk_vectors,top_k=3,min_score=0.6):
    search_results = []
    for index,chunk_vector in enumerate(chunk_vectors):
        score = float(np.dot(query_vector[0],chunk_vector))
        search_results.append({"chunk":chunks[index],"score":score})

    filtered_results = []
    for result in search_results:
        if result["score"] >= min_score:
            filtered_results.append(result)
    filtered_results.sort(key=get_score,reverse=True)
    return filtered_results[:top_k]


if __name__ == "__main__":
    model = load_embedding_model()

    documents = load_documents()
    chunks = split_documents(documents)
    chunk_vectors = build_chunk_vectors(chunks, model)

    question = input("请输入问题：")
    query_vector = model.encode([question])

    results = search_by_vector(
        query_vector=query_vector,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        top_k=len(chunks),
        min_score=-1.0,
    )

    for result in results:
        chunk = result["chunk"]
        print(chunk["source"], chunk["section"], result["score"])