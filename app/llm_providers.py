import os
import google.generativeai as genai
from groq import Groq

class LLMProvider:
    def generate_response(self, message):
        raise NotImplementedError

class GroqProvider(LLMProvider):
    def __init__(self):
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def generate_response(self, message):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model="llama3.1-70b-versatile",
        )
        return chat_completion.choices[0].message.content

class GeminiProvider(LLMProvider):
    def __init__(self):
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b-exp-0827')

    def generate_response(self, message):
        response = self.model.generate_content(message)
        return response.text
