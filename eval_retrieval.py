from rag_search import retrieve_results, merge_results
from rag_loader import load_documents, split_documents
from embedding_demo import load_embedding_model, build_chunk_vectors,search_by_vector

test_cases = [
    {
        "question": "货物寄出去以后还能调整收件地点吗？",
        "expected_source": "发货管理制度.md",
        "expected_section": "地址修改",
    },
    {
        "question": "包裹已经寄出了，还能换个地方接收吗？",
        "expected_source": "发货管理制度.md",
        "expected_section": "地址修改",
    },
]

documents = load_documents()
chunks = split_documents(documents)

embedding_model = load_embedding_model()
chunk_vectors = build_chunk_vectors(chunks, embedding_model)

hit_count = 0
for test_case in test_cases:
    print(
        f"问题：{test_case['question']}\n",
        f"预期来源：{test_case['expected_source']} / {test_case['expected_section']}\n",
    )

    question = test_case["question"]
    query_vector = embedding_model.encode([question])

    vector_results = search_by_vector(
        query_vector=query_vector,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        top_k=3,
        min_score=0.526,
    )

    keyword_results = retrieve_results(question, chunks)
    all_results = merge_results(keyword_results, vector_results)

    hit = False

    for chunk in all_results:
        if (
            chunk["source"] == test_case["expected_source"]
            and chunk["section"] == test_case["expected_section"]
        ):
            hit = True
            hit_count += 1
            break
    print("评估结果：","命中" if hit else "未命中")

    for chunk in all_results:
        print("实际召回：",chunk["source"],"/",chunk["section"])

print(f"总命中数：{hit_count}/{len(test_cases)}")