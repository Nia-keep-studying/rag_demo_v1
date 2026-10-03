from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent/"documents"
def load_documents():
    documents = []
    for file_path in DOCUMENTS_DIR.glob("*.md"):
        source = file_path.name
        content = file_path.read_text(encoding="utf-8")
        documents.append({"source":source,"content":content})
    return documents

def split_documents(documents:list):
    chunks = []

    for document in documents:
        sections = document["content"].split("## ")[1:]

        for index,section in enumerate(sections):
            heading,content = section.split("\n",1)

            chunks.append(
                {
                    "source":document["source"],
                    "section":heading.strip(),
                    "chunk_id":index,
                    "content":content.strip()
                }
            )
    return chunks

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)


    print("文档来源:",len(documents))
    print("章节数量:",len(chunks))

    for chunk in chunks:
        print(
            chunk["source"],
            chunk["section"],
            chunk["chunk_id"],
            len(chunk["content"])
        )