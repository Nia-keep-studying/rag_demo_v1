from rag_search import retrieve_results, merge_results
from rag_loader import load_documents, split_documents
from embedding_demo import load_embedding_model, build_chunk_vectors,search_by_vector
import json
from pathlib import Path

test_cases = [
    {
        "question": "货物寄出去以后还能调整收件地点吗？",
        "answerable": True,
        "expected_source": "发货管理制度.md",
        "expected_section": "地址修改",
    },
    {
        "question": "包裹已经寄出，还能换个地方接收吗？",
        "answerable": True,
        "expected_source": "发货管理制度.md",
        "expected_section": "地址修改",
    },
    {
        "question":"可以开电子发票吗？",
        "answerable":False,
        "expected_source": None,
        "expected_section": None,
    }
]

documents = load_documents()
chunks = split_documents(documents)

embedding_model = load_embedding_model()
chunk_vectors = build_chunk_vectors(chunks, embedding_model)

evaluation_results = []
pass_count = 0

for test_case in test_cases:
    print(
        f"\n问题：{test_case['question']}\n",
        f"预期来源：{test_case['expected_source']} / {test_case['expected_section']}",
    )

    question = test_case["question"]
    query_vector = embedding_model.encode([question])

    vector_results = search_by_vector(
        query_vector=query_vector,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        top_k=5,
        min_score=0.4,
    )

    keyword_results = retrieve_results(question, chunks)
    all_results = merge_results(keyword_results, vector_results)


    if test_case["answerable"]:
        hit = False

        for chunk in all_results:
            if(
                chunk["source"] == test_case["expected_source"]
                and chunk["section"] == test_case["expected_section"]
            ):
                hit = True
                break
        passed = hit

        print("评估结果：","命中" if hit else "漏召回")
    else:
        passed = not all_results
        print(
            "评估结果：",
            "无依据题返回空结果"if passed else "无依据题召回了无关片段"
        )
    if passed:
        pass_count += 1
    evaluation_results.append(
        {
            "question": question,
            "answerable": test_case["answerable"],
            "expected_source": test_case["expected_source"],
            "expected_section": test_case["expected_section"],
            "passed": passed,
            "retrieved_results": all_results,
        }
    )
    for chunk in all_results:
        print("实际召回：",chunk["source"],"/",chunk["section"])


output_path = Path(__file__).parent/"evaluation_results.json"
json_text = json.dumps(evaluation_results,ensure_ascii=False,indent=2)
output_path.write_text(json_text,encoding="utf-8")

print("评估结果已保存到：",output_path)
print(f"检索检查通过数：{pass_count}/{len(test_cases)}")
