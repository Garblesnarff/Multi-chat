import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    CEREBRAS_API_KEY = os.environ.get('CEREBRAS_API_KEY')
