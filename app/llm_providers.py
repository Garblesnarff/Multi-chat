import os
import google.generativeai as genai
from groq import Groq
from anthropic import Anthropic
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
from flask import stream_with_context, Response
import logging

logger = logging.getLogger(__name__)

class LLMProvider:
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history

    def generate_response(self, message, model):
        raise NotImplementedError

    def generate_response_with_reasoning(self, message, model):
        raise NotImplementedError

    def generate_stream(self, message, model, use_reasoning=False):
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

    def generate_response(self, message, model):
        try:
            self.add_to_history("user", message)
            chat_completion = self.client.chat.completions.create(
                messages=self.get_conversation_history(),
                model=model,
            )
            response = chat_completion.choices[0].message.content
            self.add_to_history("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Error in GroqProvider.generate_response: {str(e)}")
            raise

    def generate_response_with_reasoning(self, message, model):
        try:
            self.add_to_history("user", message)
            reasoning_prompt = f"Reason step-by-step about the following message: {message}"
            reasoning_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": reasoning_prompt}],
                model=model,
            )
            reasoning_response = reasoning_completion.choices[0].message.content

            final_prompt = f"Based on the following reasoning, provide a final response:\n\nReasoning:\n{reasoning_response}\n\nFinal response:"
            final_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                model=model,
            )
            final_response = final_completion.choices[0].message.content

            self.add_to_history("assistant", final_response)
            return f"Reasoning:\n{reasoning_response}\n\nFinal Response:\n{final_response}"
        except Exception as e:
            logger.error(f"Error in GroqProvider.generate_response_with_reasoning: {str(e)}")
            raise

    def generate_stream(self, message, model, use_reasoning=False):
        try:
            self.add_to_history("user", message)
            if use_reasoning:
                reasoning_prompt = f"Reason step-by-step about the following message: {message}"
                reasoning_stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    model=model,
                    stream=True,
                )
                final_prompt = f"Based on the reasoning, provide a final response."
                final_stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model=model,
                    stream=True,
                )
                def event_stream():
                    yield "Reasoning:\n"
                    for chunk in reasoning_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                    yield "\n\nFinal Response:\n"
                    for chunk in final_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            else:
                stream = self.client.chat.completions.create(
                    messages=self.get_conversation_history(),
                    model=model,
                    stream=True,
                )
                def event_stream():
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            return Response(stream_with_context(event_stream()), content_type='text/event-stream')
        except Exception as e:
            logger.error(f"Error in GroqProvider.generate_stream: {str(e)}")
            raise

class GeminiProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

    def generate_response(self, message, model):
        try:
            self.add_to_history("user", message)
            
            gemini_history = []
            for entry in self.get_conversation_history():
                if entry['role'] == 'user':
                    gemini_history.append({"role": "user", "parts": [{"text": entry['content']}]})
                elif entry['role'] == 'assistant':
                    gemini_history.append({"role": "model", "parts": [{"text": entry['content']}]})

            genai_model = genai.GenerativeModel(model)
            chat = genai_model.start_chat(history=gemini_history)
            response = chat.send_message(message)
            self.add_to_history("assistant", response.text)
            return response.text
        except Exception as e:
            logger.error(f"Error in GeminiProvider.generate_response: {str(e)}")
            raise

    def generate_response_with_reasoning(self, message, model):
        try:
            self.add_to_history("user", message)
            
            reasoning_prompt = f"Reason step-by-step about the following message: {message}"
            genai_model = genai.GenerativeModel(model)
            reasoning_response = genai_model.generate_content(reasoning_prompt).text

            final_prompt = f"Based on the following reasoning, provide a final response:\n\nReasoning:\n{reasoning_response}\n\nFinal response:"
            final_response = genai_model.generate_content(final_prompt).text

            self.add_to_history("assistant", final_response)
            return f"Reasoning:\n{reasoning_response}\n\nFinal Response:\n{final_response}"
        except Exception as e:
            logger.error(f"Error in GeminiProvider.generate_response_with_reasoning: {str(e)}")
            raise

    def generate_stream(self, message, model, use_reasoning=False):
        try:
            self.add_to_history("user", message)
            
            gemini_history = []
            for entry in self.get_conversation_history():
                if entry['role'] == 'user':
                    gemini_history.append({"role": "user", "parts": [{"text": entry['content']}]})
                elif entry['role'] == 'assistant':
                    gemini_history.append({"role": "model", "parts": [{"text": entry['content']}]})

            genai_model = genai.GenerativeModel(model)
            
            def event_stream():
                if use_reasoning:
                    reasoning_prompt = f"Reason step-by-step about the following message: {message}"
                    yield "Reasoning:\n"
                    for chunk in genai_model.generate_content(reasoning_prompt, stream=True):
                        if chunk.text:
                            yield chunk.text
                    
                    final_prompt = f"Based on the reasoning, provide a final response."
                    yield "\n\nFinal Response:\n"
                    for chunk in genai_model.generate_content(final_prompt, stream=True):
                        if chunk.text:
                            yield chunk.text
                else:
                    chat = genai_model.start_chat(history=gemini_history)
                    for chunk in chat.send_message(message, stream=True):
                        if chunk.text:
                            yield chunk.text
            
            return Response(stream_with_context(event_stream()), content_type='text/event-stream')
        except Exception as e:
            logger.error(f"Error in GeminiProvider.generate_stream: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        self.client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    def generate_response(self, message, model):
        try:
            self.add_to_history("user", message)
            prompt = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.get_conversation_history()])
            prompt += "\n\nAssistant:"
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens_to_sample=300
            )
            self.add_to_history("assistant", response.completion)
            return response.completion
        except Exception as e:
            logger.error(f"Error in AnthropicProvider.generate_response: {str(e)}")
            raise

    def generate_response_with_reasoning(self, message, model):
        try:
            self.add_to_history("user", message)
            reasoning_prompt = f"Reason step-by-step about the following message: {message}\n\nAssistant:"
            reasoning_response = self.client.completions.create(
                model=model,
                prompt=reasoning_prompt,
                max_tokens_to_sample=300
            ).completion

            final_prompt = f"Based on the following reasoning, provide a final response:\n\nReasoning:\n{reasoning_response}\n\nFinal response:\n\nAssistant:"
            final_response = self.client.completions.create(
                model=model,
                prompt=final_prompt,
                max_tokens_to_sample=300
            ).completion

            self.add_to_history("assistant", final_response)
            return f"Reasoning:\n{reasoning_response}\n\nFinal Response:\n{final_response}"
        except Exception as e:
            logger.error(f"Error in AnthropicProvider.generate_response_with_reasoning: {str(e)}")
            raise

    def generate_stream(self, message, model, use_reasoning=False):
        try:
            self.add_to_history("user", message)
            if use_reasoning:
                reasoning_prompt = f"Reason step-by-step about the following message: {message}\n\nAssistant:"
                reasoning_stream = self.client.completions.create(
                    model=model,
                    prompt=reasoning_prompt,
                    max_tokens_to_sample=300,
                    stream=True
                )
                final_prompt = f"Based on the reasoning, provide a final response.\n\nAssistant:"
                final_stream = self.client.completions.create(
                    model=model,
                    prompt=final_prompt,
                    max_tokens_to_sample=300,
                    stream=True
                )
                def event_stream():
                    yield "Reasoning:\n"
                    for completion in reasoning_stream:
                        if completion.completion:
                            yield completion.completion
                    yield "\n\nFinal Response:\n"
                    for completion in final_stream:
                        if completion.completion:
                            yield completion.completion
            else:
                prompt = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.get_conversation_history()])
                prompt += "\n\nAssistant:"
                stream = self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens_to_sample=300,
                    stream=True
                )
                def event_stream():
                    for completion in stream:
                        if completion.completion:
                            yield completion.completion
            return Response(stream_with_context(event_stream()), content_type='text/event-stream')
        except Exception as e:
            logger.error(f"Error in AnthropicProvider.generate_stream: {str(e)}")
            raise

class OpenAIProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    def generate_response(self, message, model):
        try:
            self.add_to_history("user", message)
            response = self.client.chat.completions.create(
                model=model,
                messages=self.get_conversation_history()
            )
            assistant_response = response.choices[0].message.content
            self.add_to_history("assistant", assistant_response)
            return assistant_response
        except Exception as e:
            logger.error(f"Error in OpenAIProvider.generate_response: {str(e)}")
            raise

    def generate_response_with_reasoning(self, message, model):
        try:
            self.add_to_history("user", message)
            reasoning_prompt = f"Reason step-by-step about the following message: {message}"
            reasoning_response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": reasoning_prompt}]
            ).choices[0].message.content

            final_prompt = f"Based on the following reasoning, provide a final response:\n\nReasoning:\n{reasoning_response}\n\nFinal response:"
            final_response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}]
            ).choices[0].message.content

            self.add_to_history("assistant", final_response)
            return f"Reasoning:\n{reasoning_response}\n\nFinal Response:\n{final_response}"
        except Exception as e:
            logger.error(f"Error in OpenAIProvider.generate_response_with_reasoning: {str(e)}")
            raise

    def generate_stream(self, message, model, use_reasoning=False):
        try:
            self.add_to_history("user", message)
            if use_reasoning:
                reasoning_prompt = f"Reason step-by-step about the following message: {message}"
                reasoning_stream = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    stream=True
                )
                final_prompt = f"Based on the reasoning, provide a final response."
                final_stream = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": final_prompt}],
                    stream=True
                )
                def event_stream():
                    yield "Reasoning:\n"
                    for chunk in reasoning_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                    yield "\n\nFinal Response:\n"
                    for chunk in final_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            else:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=self.get_conversation_history(),
                    stream=True
                )
                def event_stream():
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            return Response(stream_with_context(event_stream()), content_type='text/event-stream')
        except Exception as e:
            logger.error(f"Error in OpenAIProvider.generate_stream: {str(e)}")
            raise

class CerebrasProvider(LLMProvider):
    def __init__(self, max_history=10):
        super().__init__(max_history)
        self.client = Cerebras(api_key=os.environ.get('CEREBRAS_API_KEY'))

    def generate_response(self, message, model):
        try:
            self.add_to_history("user", message)
            chat_completion = self.client.chat.completions.create(
                messages=self.get_conversation_history(),
                model=model,
            )
            response = chat_completion.choices[0].message.content
            self.add_to_history("assistant", response)
            return response
        except Exception as e:
            logger.error(f"Error in CerebrasProvider.generate_response: {str(e)}")
            raise

    def generate_response_with_reasoning(self, message, model):
        try:
            self.add_to_history("user", message)
            reasoning_prompt = f"Reason step-by-step about the following message: {message}"
            reasoning_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": reasoning_prompt}],
                model=model,
            )
            reasoning_response = reasoning_completion.choices[0].message.content

            final_prompt = f"Based on the following reasoning, provide a final response:\n\nReasoning:\n{reasoning_response}\n\nFinal response:"
            final_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                model=model,
            )
            final_response = final_completion.choices[0].message.content

            self.add_to_history("assistant", final_response)
            return f"Reasoning:\n{reasoning_response}\n\nFinal Response:\n{final_response}"
        except Exception as e:
            logger.error(f"Error in CerebrasProvider.generate_response_with_reasoning: {str(e)}")
            raise

    def generate_stream(self, message, model, use_reasoning=False):
        try:
            self.add_to_history("user", message)
            if use_reasoning:
                reasoning_prompt = f"Reason step-by-step about the following message: {message}"
                reasoning_stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    model=model,
                    stream=True,
                )
                final_prompt = f"Based on the reasoning, provide a final response."
                final_stream = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": final_prompt}],
                    model=model,
                    stream=True,
                )
                def event_stream():
                    yield "Reasoning:\n"
                    for chunk in reasoning_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                    yield "\n\nFinal Response:\n"
                    for chunk in final_stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            else:
                stream = self.client.chat.completions.create(
                    messages=self.get_conversation_history(),
                    model=model,
                    stream=True,
                )
                def event_stream():
                    for chunk in stream:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
            return Response(stream_with_context(event_stream()), content_type='text/event-stream')
        except Exception as e:
            logger.error(f"Error in CerebrasProvider.generate_stream: {str(e)}")
            raise