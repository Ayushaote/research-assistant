import google.generativeai as genai
import os

# Replace with your real key
genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro")

response = model.generate_content("Explain the main challenges of segmenting surgical instruments in laparoscopic videos.")
print(response.text)
