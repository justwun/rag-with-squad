from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """Làm sạch text: bỏ ký tự thừa, khoảng trắng"""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text


def preprocess_squad(split: str = "train[:2000]"):
    """Load và xử lý dataset SQuAD"""
    dataset = load_dataset("rajpurkar/squad", split=split)

    docs = []
    for example in dataset:
        context = clean_text(example["context"])
        if context.strip():
            docs.append(Document(page_content=context))

    # Chunking để phù hợp cho embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)

    return chunked_docs


if __name__ == "__main__":
    docs = preprocess_squad()
    print(f"✅ Preprocessing done. Got {len(docs)} documents after cleaning + chunking.")
    print("Sample doc:", docs[0])
