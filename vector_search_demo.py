import numpy as np

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return dot_product / (norm_a * norm_b)

query_vector = np.array([1.0, 0.0])

document_vectors = {
    "地址修改": np.array([0.9, 0.1]),
    "退款规则": np.array([0.2, 0.8]),
    "完全无关": np.array([0.0, 1.0])
}

# search_results = []
# for name, document_vector in document_vectors.items():
#     score = cosine_similarity(query_vector, document_vector)

#     search_results.append({
#         "name":name,
#         "score":float(score)
#     })

def get_score(result):
    return result["score"]

# search_results.sort(key=get_score,reverse=True)

# min_score = 0.5
# filtered_results = []

# for result in search_results:
#     if result["score"] >= min_score:
#         filtered_results.append(result)

# top_k = 2
# top_results = filtered_results[:top_k]

# for result in top_results:
#     print(result["name"],result["score"])


def search_by_vector(query_vector,document_vectors,top_k=2,min_score=0.5):
    search_results = []
    for name, document_vector in document_vectors.items():
        score = cosine_similarity(query_vector, document_vector)

        search_results.append({
            "name":name,
            "score":float(score)
        })

    search_results.sort(key=get_score,reverse=True)
    filtered_results = [result for result in search_results if result["score"] >= min_score]
    return filtered_results[:top_k]

top_results = search_by_vector(
    query_vector=query_vector,
    document_vectors=document_vectors,
    top_k=2,
    min_score=0.5
)

for result in top_results:
    print(result["name"], result["score"])