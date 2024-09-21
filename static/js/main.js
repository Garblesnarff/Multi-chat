console.log('main.js loaded');

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded event fired');
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const providerSelects = document.querySelectorAll('.provider-select');
    const comparisonContainer = document.getElementById('comparison-container');
    const reasoningCheckbox = document.getElementById('reasoning-checkbox');
    const streamingCheckbox = document.getElementById('streaming-checkbox');

    function addMessage(content, isUser = false, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', isUser ? 'user-message' : 'bot-message');
        if (isError) {
            messageDiv.classList.add('error-message');
        }
        messageDiv.textContent = content;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function getSelectedProviders() {
        const selectedProviders = {};
        providerSelects.forEach(select => {
            if (select.value) {
                const provider = select.id.split('-')[0];
                selectedProviders[provider] = select.value;
            }
        });
        return selectedProviders;
    }

    function displayComparison(responses) {
        comparisonContainer.innerHTML = '';
        comparisonContainer.classList.remove('hidden');

        Object.entries(responses).forEach(([provider, response]) => {
            const providerDiv = document.createElement('div');
            providerDiv.classList.add('mb-4');
            providerDiv.innerHTML = `
                <h3 class="font-bold text-lg mb-2">${provider.charAt(0).toUpperCase() + provider.slice(1)}</h3>
                <p class="bg-white rounded p-2 ${response.startsWith('Error:') ? 'text-red-500' : ''}">${response}</p>
            `;
            comparisonContainer.appendChild(providerDiv);
        });
    }

    function displayStreamingResponse(provider, content) {
        let providerDiv = document.getElementById(`streaming-${provider}`);
        if (!providerDiv) {
            providerDiv = document.createElement('div');
            providerDiv.id = `streaming-${provider}`;
            providerDiv.classList.add('mb-4');
            providerDiv.innerHTML = `
                <h3 class="font-bold text-lg mb-2">${provider.charAt(0).toUpperCase() + provider.slice(1)}</h3>
                <p class="bg-white rounded p-2"></p>
            `;
            comparisonContainer.appendChild(providerDiv);
        }
        const responseP = providerDiv.querySelector('p');
        responseP.textContent += content;
        comparisonContainer.scrollTop = comparisonContainer.scrollHeight;
    }

    async function sendMessage() {
        console.log('sendMessage function called');
        const message = userInput.value.trim();
        console.log('User input:', message);
        const selectedProviders = getSelectedProviders();
        const useReasoning = reasoningCheckbox && reasoningCheckbox.checked;
        const useStreaming = streamingCheckbox && streamingCheckbox.checked;

        if (message && Object.keys(selectedProviders).length > 0) {
            addMessage(message, true);
            userInput.value = '';

            if (useStreaming) {
                comparisonContainer.innerHTML = '';
                comparisonContainer.classList.remove('hidden');
                const eventSource = new EventSource(`/chat?message=${encodeURIComponent(message)}&providers=${encodeURIComponent(JSON.stringify(selectedProviders))}&use_reasoning=${useReasoning}&use_streaming=true`);
                
                let currentProvider = '';

                eventSource.onmessage = function(event) {
                    if (event.data === '[DONE]') {
                        eventSource.close();
                    } else if (currentProvider === '') {
                        currentProvider = event.data;
                    } else {
                        displayStreamingResponse(currentProvider, event.data);
                    }
                };

                eventSource.onerror = function(event) {
                    console.error('EventSource failed:', event);
                    eventSource.close();
                    addMessage('Error: Unable to get a streaming response from the server.', false, true);
                };
            } else {
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ 
                            message, 
                            providers: selectedProviders, 
                            use_reasoning: useReasoning, 
                            use_streaming: useStreaming 
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.error) {
                            addMessage(`Error: ${data.error}`, false, true);
                        } else {
                            displayComparison(data.responses);
                        }
                    } else {
                        const errorData = await response.json();
                        addMessage(`Error: ${errorData.error || 'Unable to get a response from the server.'}`, false, true);
                    }
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('Error: Unable to connect to the server.', false, true);
                }
            }
        } else if (Object.keys(selectedProviders).length === 0) {
            addMessage('Please select at least one provider and model.', false, true);
        }
    }

    async function clearHistory() {
        const selectedProviders = getSelectedProviders();
        for (const provider of Object.keys(selectedProviders)) {
            try {
                const response = await fetch('/clear_history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ provider }),
                });

                if (response.ok) {
                    console.log(`Cleared history for ${provider}`);
                } else {
                    console.error(`Failed to clear history for ${provider}`);
                    const errorData = await response.json();
                    addMessage(`Error clearing history for ${provider}: ${errorData.error}`, false, true);
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage(`Error clearing history for ${provider}: ${error.message}`, false, true);
            }
        }

        chatContainer.innerHTML = '';
        comparisonContainer.innerHTML = '';
        comparisonContainer.classList.add('hidden');
        addMessage('Conversation history cleared.');
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', () => {
            console.log('Send button clicked');
            sendMessage();
        });
    }
    
    if (userInput) {
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                console.log('Enter key pressed');
                sendMessage();
            }
        });
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearHistory);
    }
});
