import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question, mode):
    if mode == "Summarize":
        prompt = f"Summarize the following research content:\n\n{context}"
    elif mode == "Explain":
        prompt = f"Explain the key ideas in this passage for better understanding:\n\n{context}\n\nExplanation:"
    elif mode == "Q&A":
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"{context}\n\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    output = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)


with gr.Blocks() as demo:
    gr.Markdown("## 🤖 LLM-Powered Research Assistant")
    with gr.Row():
        with gr.Column():
            context = gr.Textbox(lines=20, label="Context (Paste research paper content or abstract)")
            question = gr.Textbox(label="Question", placeholder="e.g., Explain this intro / What is the main idea?")
            mode = gr.Dropdown(choices=["Explain", "Summarize", "Extract Contributions", "Extract Limitations"], value="Explain", label="Mode")
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Answer")
            share = gr.Button("Share via Link")

    submit_btn.click(generate_answer, inputs=[context, question, mode], outputs=output)
    share.click(generate_answer, inputs=[context, question, mode], outputs=output)

demo.launch()
