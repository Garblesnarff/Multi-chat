import os
import google.generativeai as genai
from groq import Groq
from anthropic import Anthropic
import openai

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

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    def generate_response(self, message):
        response = self.client.completions.create(
            model="claude-2",
            prompt=f"\n\nHuman: {message}\n\nAssistant:",
            max_tokens_to_sample=300
        )
        return response.completion

class OpenAIProvider(LLMProvider):
    def __init__(self):
        openai.api_key = os.environ.get('OPENAI_API_KEY')

    def generate_response(self, message):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content
