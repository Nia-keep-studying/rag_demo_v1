import json
from pathlib import Path
from sentence_transformers import CrossEncoder

input_path = Path(__file__).parent/"evaluation_results.json"
evaluation_results = json.loads(input_path.read_text(encoding="utf-8"))

test_result = evaluation_results[1]
question = test_result["question"]
chunks = test_result["retrieved_results"]

pairs = []
for chunk in chunks:
    text = f"章节:{chunk['section']}\n正文:{chunk['content']}"
    pairs.append([question,text])

reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    device = "cpu",
    cache_folder=str(Path(__file__).parent/".model_cache"),
)

scores = reranker.predict(pairs)

rerank_results = []

for index,score in enumerate(scores):
    rerank_results.append(
        {
            "chunk":chunks[index],
            "score":float(score),
        }
    )

def get_rerank_score(result):
    return result["score"]

rerank_results.sort(key=get_rerank_score,reverse=True)

final_top_k = 2
selected_results = rerank_results[:final_top_k]

print("最终选中的片段：")
for result in selected_results:
    print(result["chunk"]["section"])

for result in rerank_results:
    print(result["chunk"]["section"],result["score"])

# for index,score in enumerate(scores):
#     print(
#         question,
#         chunks[index]["section"],
#         float(score)
#     )