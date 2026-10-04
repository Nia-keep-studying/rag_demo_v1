from rag_loader import load_documents,split_documents
import json

# documents = load_documents()
# chunks = split_documents(documents)



def search_by_keyword(chunks,keyword):
    result = []

    for chunk in chunks:
        content = f"{chunk["source"]},{chunk["section"]},{chunk["content"]}"
        if keyword in content:
            result.append(chunk)

    return result

def build_context(results):
    context = []
    for result in results:
        source = result["source"]
        section = result["section"]
        content = result["content"]
        context.append({"来源":source,"章节":section,"正文":content})
    return json.dumps(context,ensure_ascii=False,indent=2)

def extract_keywords(question):
    keyword_mapping = {
    "地址":["地址","收件地址","收货地点","收货地址","收件地点"],
    "退款":["退款","退钱"],
    "取消":["取消","撤销订单"]
    }

    matched_keywords = []
    for keyword,variants in keyword_mapping.items():
        for variant in variants:
            if variant in question:
                matched_keywords.append(keyword)
                break
    return matched_keywords

def retrieve_results(question,chunks):
    keywords = extract_keywords(question)
    all_results = []
    for keyword in keywords:
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

def merge_results(keyword_results,vector_results):
    results = []
    for keyword_result in keyword_results:
        results.append(keyword_result)
    for vector_result in vector_results:
        results.append(vector_result["chunk"])

    unique_results = []
    seen = set()

    for result in results:
        result_id = (result["source"],result["chunk_id"])
        if result_id not in seen:
            unique_results.append(result)
            seen.add(result_id)
    return unique_results



if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    keyword = input("请输入检索关键词：")
    results = extract_keywords(keyword)

    print(results)
