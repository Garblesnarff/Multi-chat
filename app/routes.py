from flask import Blueprint, render_template, request, jsonify, session, Response, stream_with_context
from app.llm_providers import GroqProvider, GeminiProvider, AnthropicProvider, OpenAIProvider, CerebrasProvider
import logging
import json

bp = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/chat', methods=['POST', 'GET'])
def chat():
    try:
        if request.method == 'GET':
            # Handle streaming request
            message = request.args.get('message')
            providers = json.loads(request.args.get('providers'))
            use_reasoning = request.args.get('use_reasoning') == 'true'
            use_streaming = request.args.get('use_streaming') == 'true'
        else:
            # Handle non-streaming request
            data = request.json
            message = data.get('message')
            providers = data.get('providers', {})
            use_reasoning = data.get('use_reasoning', False)
            use_streaming = data.get('use_streaming', False)

        logger.debug(f"Received chat request: message={message}, providers={providers}, use_reasoning={use_reasoning}, use_streaming={use_streaming}")

        if 'llm_provider' not in session:
            session['llm_provider'] = {}

        if use_streaming:
            def generate():
                try:
                    for provider, model in providers.items():
                        yield f"data: {provider}\n\n"
                        llm = get_llm_provider(provider)
                        for chunk in llm.generate_stream(message, model, use_reasoning):
                            yield f"data: {chunk}\n\n"
                        yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error in generate function: {str(e)}")
                    yield f"data: Error: {str(e)}\n\n"
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            responses = {}
            for provider, model in providers.items():
                llm = get_llm_provider(provider)
                try:
                    if use_reasoning:
                        responses[provider] = llm.generate_response_with_reasoning(message, model)
                    else:
                        responses[provider] = llm.generate_response(message, model)
                    session['llm_provider'][provider] = llm.to_dict()
                except Exception as e:
                    logger.error(f"Error generating response for provider {provider}: {str(e)}")
                    responses[provider] = f"Error: {str(e)}"
            
            return jsonify({'responses': responses})
    except Exception as e:
        logger.error(f"Unexpected error in chat route: {str(e)}")
        if use_streaming:
            def generate():
                yield f"data: Error: {str(e)}\n\n"
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return jsonify({'error': str(e)}), 500

@bp.route('/clear_history', methods=['POST'])
def clear_history():
    data = request.json
    provider = data.get('provider')

    if 'llm_provider' in session and provider in session['llm_provider']:
        session['llm_provider'][provider] = get_llm_provider(provider, new_instance=True).to_dict()
        return jsonify({'message': 'Conversation history cleared'}), 200
    else:
        return jsonify({'error': 'Invalid provider or no conversation history'}), 400

def get_llm_provider(provider, new_instance=False):
    if new_instance or provider not in session.get('llm_provider', {}):
        if provider == 'groq':
            return GroqProvider()
        elif provider == 'gemini':
            return GeminiProvider()
        elif provider == 'anthropic':
            return AnthropicProvider()
        elif provider == 'openai':
            return OpenAIProvider()
        elif provider == 'cerebras':
            return CerebrasProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    else:
        llm_dict = session['llm_provider'][provider]
        if provider == 'groq':
            return GroqProvider.from_dict(llm_dict)
        elif provider == 'gemini':
            return GeminiProvider.from_dict(llm_dict)
        elif provider == 'anthropic':
            return AnthropicProvider.from_dict(llm_dict)
        elif provider == 'openai':
            return OpenAIProvider.from_dict(llm_dict)
        elif provider == 'cerebras':
            return CerebrasProvider.from_dict(llm_dict)
