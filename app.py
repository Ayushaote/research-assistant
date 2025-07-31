import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
model_name = "google/flan-t5-base"  # or any other flan-t5 variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question):
    prompt = (
        "You are a helpful AI assistant. Use the provided research paper context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# Gradio UI
iface = gr.Interface(
    fn=generate_answer,
    inputs=[
        gr.Textbox(label="Context (Paste research paper content or abstract)", lines=10, placeholder="Paste your research paper content here..."),
        gr.Textbox(label="Question", placeholder="e.g., What is the main contribution?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="LLM-Powered Research Assistant",
    description="Summarize or extract insights from research papers using an open-source FLAN-T5 model"
)

if __name__ == "__main__":
    iface.launch()
