import google.generativeai as genai

API_KEY = "TA_CLE_ICI"

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content("Bonjour")

print(response.text)