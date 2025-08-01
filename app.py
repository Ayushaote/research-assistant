import gradio as gr
import google.generativeai as genai
import os

# ✅ Replace with your actual Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or "AIzaSyBW1cJuL8EDmQzussOyqKQOkfH78PlB6ow")

# ✅ Load the best free-supported model
try:
    model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as e:
    raise RuntimeError(f"Failed to load Gemini model: {e}")

# ✅ Response generator
def generate_answer(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {e}"

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 LLM-Powered Research Assistant (Gemini)")
    with gr.Row():
        input_box = gr.Textbox(placeholder="Ask a research question...", label="Your Question")
    output_box = gr.Textbox(label="Gemini's Response")

    submit_btn = gr.Button("Generate Answer")

    submit_btn.click(fn=generate_answer, inputs=[input_box], outputs=[output_box])

# ✅ Run the app
if __name__ == "__main__":
    demo.launch()
