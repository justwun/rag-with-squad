import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

# Cấu hình
INDEX_DIR = "faiss_squad_index"
# Lấy API key
api_key = os.getenv("GOOGLE_API_KEY")

# Cấu hình Gemini
genai.configure(api_key=api_key)

# Load FAISS + embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def rag_answer(question):
    # 1. Retrieve context
    docs = db.similarity_search(question, k=3)
    retrieved = [d.page_content for d in docs]

    # 2. Prompt
    prompt = f"""
    You are a helpful assistant. 
    Use the following context if it is relevant. 
    If the context does not help, answer from your own knowledge.

    Context: {retrieved}
    Question: {question}
    Answer:
    """

    # 3. Call Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text, "\n\n---\n\n".join(retrieved)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📚 RAG Demo with SQuAD + FAISS + Gemini")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Enter your question", placeholder="Ask me anything from SQuAD data...")
            btn = gr.Button("Get Answer")
        with gr.Column(scale=3):
            answer = gr.Textbox(label="Answer")
            context = gr.Textbox(label="Retrieved Context", lines=8)

    btn.click(fn=rag_answer, inputs=question, outputs=[answer, context])

if __name__ == "__main__":
    demo.launch()
