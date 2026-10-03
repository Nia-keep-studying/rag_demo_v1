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

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    keyword = input("请输入检索关键词：")
    results = search_by_keyword(chunks, keyword)

    print(build_context(results=results))
