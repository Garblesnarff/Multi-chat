from flask import Blueprint, render_template, request, jsonify, session
from app.llm_providers import GroqProvider, GeminiProvider, AnthropicProvider, OpenAIProvider, CerebrasProvider

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    providers = data.get('providers', {})
    use_reasoning = data.get('use_reasoning', False)

    if 'llm_provider' not in session:
        session['llm_provider'] = {}

    responses = {}
    for provider, model in providers.items():
        if provider not in session['llm_provider']:
            if provider == 'groq':
                session['llm_provider'][provider] = GroqProvider().to_dict()
            elif provider == 'gemini':
                session['llm_provider'][provider] = GeminiProvider().to_dict()
            elif provider == 'anthropic':
                session['llm_provider'][provider] = AnthropicProvider().to_dict()
            elif provider == 'openai':
                session['llm_provider'][provider] = OpenAIProvider().to_dict()
            elif provider == 'cerebras':
                session['llm_provider'][provider] = CerebrasProvider().to_dict()
            else:
                continue

        llm_dict = session['llm_provider'][provider]
        if provider == 'groq':
            llm = GroqProvider.from_dict(llm_dict)
        elif provider == 'gemini':
            llm = GeminiProvider.from_dict(llm_dict)
        elif provider == 'anthropic':
            llm = AnthropicProvider.from_dict(llm_dict)
        elif provider == 'openai':
            llm = OpenAIProvider.from_dict(llm_dict)
        elif provider == 'cerebras':
            llm = CerebrasProvider.from_dict(llm_dict)
        
        if use_reasoning:
            responses[provider] = llm.generate_response_with_reasoning(message, model)
        else:
            responses[provider] = llm.generate_response(message, model)
        session['llm_provider'][provider] = llm.to_dict()

    return jsonify({'responses': responses})

@bp.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    provider = data.get('provider')

    if 'llm_provider' in session and provider in session['llm_provider']:
        if provider == 'groq':
            session['llm_provider'][provider] = GroqProvider().to_dict()
        elif provider == 'gemini':
            session['llm_provider'][provider] = GeminiProvider().to_dict()
        elif provider == 'anthropic':
            session['llm_provider'][provider] = AnthropicProvider().to_dict()
        elif provider == 'openai':
            session['llm_provider'][provider] = OpenAIProvider().to_dict()
        elif provider == 'cerebras':
            session['llm_provider'][provider] = CerebrasProvider().to_dict()
        return jsonify({'message': 'Conversation history cleared'}), 200
    else:
        return jsonify({'error': 'Invalid provider or no conversation history'}), 400
