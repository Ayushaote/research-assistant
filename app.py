import gradio as gr
import google.generativeai as genai
import os

# Set your API key
genai.configure(api_key="AIzaSyBW1cJuL8EDmQzussOyqKQOkfH78PlB6ow")

model = genai.GenerativeModel("gemini-pro")

def generate_answer(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=5, placeholder="Ask something about the paper..."),
    outputs="text",
    title="LLM Research Assistant (Gemini)",
    description="Ask questions or summarize academic text using Gemini Pro"
)

iface.launch()
