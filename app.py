import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer
model_name = "google/flan-t5-large"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(prompt):
    output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return output[0]["generated_text"]

# Gradio Interface
demo = gr.Interface(fn=generate_response, inputs="text", outputs="text", title="LLM Research Assistant")
demo.launch()
