import os
import google.generativeai as genai
from groq import Groq
from anthropic import Anthropic
from openai import OpenAI

class LLMProvider:
    def __init__(self):
        self.conversation_history = []

    def generate_response(self, message):
        raise NotImplementedError

    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})

    def get_conversation_history(self):
        return self.conversation_history

class GroqProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def generate_response(self, message):
        self.add_to_history("user", message)
        chat_completion = self.client.chat.completions.create(
            messages=self.get_conversation_history(),
            model="llama3.1-70b-versatile",
        )
        response = chat_completion.choices[0].message.content
        self.add_to_history("assistant", response)
        return response

class GeminiProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b-exp-0827')

    def generate_response(self, message):
        self.add_to_history("user", message)
        chat = self.model.start_chat(history=self.get_conversation_history())
        response = chat.send_message(message)
        self.add_to_history("assistant", response.text)
        return response.text

class AnthropicProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    def generate_response(self, message):
        self.add_to_history("user", message)
        prompt = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.get_conversation_history()])
        prompt += "\n\nAssistant:"
        response = self.client.completions.create(
            model="claude-2",
            prompt=prompt,
            max_tokens_to_sample=300
        )
        self.add_to_history("assistant", response.completion)
        return response.completion

class OpenAIProvider(LLMProvider):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def generate_response(self, message):
        self.add_to_history("user", message)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.get_conversation_history()
        )
        assistant_response = response.choices[0].message.content
        self.add_to_history("assistant", assistant_response)
        return assistant_response
