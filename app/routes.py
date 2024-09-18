from flask import Blueprint, render_template, request, jsonify, session
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

    if 'llm_provider' not in session:
        session['llm_provider'] = {}

    if provider not in session['llm_provider']:
        if provider == 'groq':
            session['llm_provider'][provider] = GroqProvider()
        elif provider == 'gemini':
            session['llm_provider'][provider] = GeminiProvider()
        elif provider == 'anthropic':
            session['llm_provider'][provider] = AnthropicProvider()
        elif provider == 'openai':
            session['llm_provider'][provider] = OpenAIProvider()
        else:
            return jsonify({'error': 'Invalid provider'}), 400

    llm = session['llm_provider'][provider]
    response = llm.generate_response(message)
    return jsonify({'response': response})

@bp.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    provider = data.get('provider')

    if 'llm_provider' in session and provider in session['llm_provider']:
        session['llm_provider'][provider].conversation_history = []
        return jsonify({'message': 'Conversation history cleared'}), 200
    else:
        return jsonify({'error': 'Invalid provider or no conversation history'}), 400
