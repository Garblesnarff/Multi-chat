import os
import google.generativeai as genai
from google.generativeai.types import content_types
from groq import Groq
from anthropic import Anthropic
from openai import OpenAI

class LLMProvider:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history

    def generate_response(self, message):
        raise NotImplementedError

    def add_to_history(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_conversation_history(self):
        return self.conversation_history

    def to_dict(self):
        return {
            "max_history": self.max_history,
            "conversation_history": self.conversation_history
        }

    @classmethod
    def from_dict(cls, data):
        provider = cls(max_history=data.get("max_history", 10))
        provider.conversation_history = data.get("conversation_history", [])
        return provider

class GroqProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        self.client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def generate_response(self, message):
        self.add_to_history("user", message)
        chat_completion = self.client.chat.completions.create(
            messages=self.get_conversation_history(),
            model="mixtral-8x7b-32768",  # Updated to a valid Groq model
        )
        response = chat_completion.choices[0].message.content
        self.add_to_history("assistant", response)
        return response

class GeminiProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b-exp-0827')

    def generate_response(self, message):
        self.add_to_history("user", message)
        
        # Convert conversation history to Gemini-compatible format
        gemini_history = []
        for entry in self.get_conversation_history():
            if entry['role'] == 'user':
                gemini_history.append(content_types.Content(role='user', parts=[content_types.Part(text=entry['content'])]))
            elif entry['role'] == 'assistant':
                gemini_history.append(content_types.Content(role='model', parts=[content_types.Part(text=entry['content'])]))

        chat = self.model.start_chat(history=gemini_history)
        response = chat.send_message(message)
        self.add_to_history("assistant", response.text)
        return response.text

class AnthropicProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
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
    def __init__(self, max_history=10):
        super().__init__(max_history)
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
