from flask import Blueprint, render_template, request, jsonify
from app.llm_providers import GroqProvider, GeminiProvider, AnthropicProvider, OpenAIProvider

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    provider = data.get('provider')

    if provider == 'groq':
        llm = GroqProvider()
    elif provider == 'gemini':
        llm = GeminiProvider()
    elif provider == 'anthropic':
        llm = AnthropicProvider()
    elif provider == 'openai':
        llm = OpenAIProvider()
    else:
        return jsonify({'error': 'Invalid provider'}), 400

    response = llm.generate_response(message)
    return jsonify({'response': response})
