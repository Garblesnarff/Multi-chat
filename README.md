# EchoChat: Multi-Provider LLM Chatbot

EchoChat is a multi-provider LLM chatbot with a clean, modern dashboard using Flask and Vanilla JS.

## Features:

- [x] Using a unified interface to try out different providers
- [x] Configuring the app from the sidebar
- [x] Comparing responses from different models

## Providers:

- [x] Groq (remote, requires API key)
- [x] Gemini (remote, requires API key)
- [x] Anthropic (remote, requires API key)
- [x] OpenAI (remote, requires API key)
- [x] Cerebras (remote, requires API key)

## Work in progress:

- [ ] Implement streaming output for LLM responses

## Getting Started

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables (API keys for different providers)
4. Run the application:
   ```
   python main.py
   ```

## Usage

1. Select the desired LLM providers and models
2. Type your message in the input field
3. Click "Send" or press Enter to get responses from the selected models
4. Compare the responses in the comparison container

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
