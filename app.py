import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fitz  # PyMuPDF
import torch

# Load model and tokenizer
model_name = "MBZUAI/LaMini-Flan-T5-783M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Generate answer from prompt
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# Full pipeline
def process_pdf(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF."

    text = extract_text_from_pdf(pdf_file)

    # Create prompts
    summary_prompt = f"summarize: {text[:1500]}"
    keypoints_prompt = f"extract key points: {text[:1500]}"
    question_prompt = f"answer the question: {question} based on this context: {text[:1500]}"

    # Get outputs
    summary = generate_answer(summary_prompt)
    keypoints = generate_answer(keypoints_prompt)
    answer = generate_answer(question_prompt)

    return summary, keypoints, answer

# Gradio UI
iface = gr.Interface(
    fn=process_pdf,
    inputs=[
        gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"]),
        gr.Textbox(label="Ask a Question About the Paper")
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Key Points"),
        gr.Textbox(label="Answer")
    ],
    title="LLM-powered Research Assistant",
    description="Upload a PDF research paper, get a summary, key points, and ask questions using an open-source model."
)

if __name__ == "__main__":
    iface.launch()
