import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question, mode):
    instruction = {
        "Summarize": "Summarize the following context in simple terms.",
        "Explain": "Explain the following research content in detail.",
        "Extract Contributions": "List the main contributions from the research paper.",
        "Extract Limitations": "List any limitations or challenges mentioned in the paper."
    }.get(mode, "Answer the question based on the context.")

    prompt = f"{instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
