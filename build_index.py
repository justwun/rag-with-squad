import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from data_preprocessing import preprocess_squad

INDEX_DIR = "faiss_squad_index"


def build_index():
    docs = preprocess_squad()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    print(f"✅ Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
