from google import genai

client = genai.Client(api_key="AIzaSyBzCMnDmfR9TLyvTBchKM6frGnHd3nxMHk")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=""
)
print(response.text)
