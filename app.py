import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question, mode):
    if mode == "Summarize":
        prompt = f"Summarize the following research content:\n\n{context}"
    elif mode == "Explain":
        prompt = f"Explain the key ideas in this passage:\n\n{context}\n\nExplanation:"
    elif mode == "Q&A":
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"{context}\n\n{question}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    output = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def clear_inputs():
    return "", "", "Summarize", ""

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 LLM Research Assistant")
    with gr.Row():
        with gr.Column():
            context_input = gr.Textbox(lines=15, label="📄 Paper Abstract / Section / Context")
            question_input = gr.Textbox(lines=2, label="❓ Question (Optional for Summarize/Explain)")
            mode_input = gr.Radio(choices=["Summarize", "Explain", "Q&A"], value="Summarize", label="Select Mode")
            submit_btn = gr.Button("🚀 Generate")
            clear_btn = gr.Button("Clear")
        with gr.Column():
            output_text = gr.Textbox(label="🧠 Output", lines=20)

    submit_btn.click(fn=generate_answer, inputs=[context_input, question_input, mode_input], outputs=output_text)
    clear_btn.click(fn=clear_inputs, outputs=[context_input, question_input, mode_input, output_text])

if __name__ == "__main__":
    demo.launch()
