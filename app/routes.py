from flask import Blueprint, render_template, request, jsonify, session, Response, stream_with_context
from app.llm_providers import GroqProvider, GeminiProvider, AnthropicProvider, OpenAIProvider, CerebrasProvider
import logging

bp = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        providers = data.get('providers', {})
        use_reasoning = data.get('use_reasoning', False)
        use_streaming = data.get('use_streaming', False)

        logger.debug(f"Received chat request: message={message}, providers={providers}, use_reasoning={use_reasoning}, use_streaming={use_streaming}")

        if 'llm_provider' not in session:
            session['llm_provider'] = {}

        responses = {}
        for provider, model in providers.items():
            logger.debug(f"Processing provider: {provider}")
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
                    logger.warning(f"Unknown provider: {provider}")
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
            
            try:
                if use_streaming:
                    responses[provider] = llm.generate_stream(message, model)
                else:
                    if use_reasoning:
                        responses[provider] = llm.generate_response_with_reasoning(message, model)
                    else:
                        responses[provider] = llm.generate_response(message, model)
                session['llm_provider'][provider] = llm.to_dict()
            except Exception as e:
                logger.error(f"Error generating response for provider {provider}: {str(e)}")
                responses[provider] = f"Error: {str(e)}"

        if use_streaming:
            def generate():
                for provider, response in responses.items():
                    yield f"data: {provider}\n\n"
                    for chunk in response.response:
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return jsonify({'responses': responses})
    except Exception as e:
        logger.error(f"Unexpected error in chat route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

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
