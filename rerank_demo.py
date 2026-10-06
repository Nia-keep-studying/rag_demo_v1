import os
import json
from pathlib import Path

# 在导入模型相关库之前指定缓存，供嵌入模型和重排序模型共同使用。
os.environ["HF_HOME"] = str(Path(__file__).parent / ".model_cache")
from sentence_transformers import CrossEncoder

def load_rerank_model():
    reranker = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        device = "cpu",
        cache_folder=str(Path(__file__).parent/".model_cache"),
    )
    return reranker

def get_rerank_score(result):
    return result["score"]

def rerank_chunks(question,chunks,reranker,top_k=2):
    if not chunks:
        return []
    pairs = []
    for chunk in chunks:
        text = f"章节:{chunk['section']}\n正文:{chunk['content']}"
        pairs.append([question,text])

    scores = reranker.predict(pairs)

    rerank_results = []

    for index,score in enumerate(scores):
        rerank_results.append(
            {
                "chunk":chunks[index],
                "score":float(score),
            }
        )

    rerank_results.sort(key=get_rerank_score,reverse=True)

    selected_results = rerank_results[:top_k]

    return selected_results

if __name__ == "__main__":
    input_path = Path(__file__).parent/"evaluation_results.json"
    evaluation_results = json.loads(input_path.read_text(encoding="utf-8"))

    test_result = evaluation_results[1]
    question = test_result["question"]
    chunks = test_result["retrieved_results"]

    reranker = load_rerank_model()
    selected_results = rerank_chunks(question,chunks,reranker,top_k=5)

    print("最终选中的片段：")
    for result in selected_results:
        print(result["chunk"]["section"],result["score"])
