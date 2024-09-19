document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const providerSelects = document.querySelectorAll('.provider-select');
    const comparisonContainer = document.getElementById('comparison-container');

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', isUser ? 'user-message' : 'bot-message');
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
                <p class="bg-white rounded p-2">${response}</p>
            `;
            comparisonContainer.appendChild(providerDiv);
        });
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        const selectedProviders = getSelectedProviders();

        if (message && Object.keys(selectedProviders).length > 0) {
            addMessage(message, true);
            userInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message, providers: selectedProviders }),
                });

                if (response.ok) {
                    const data = await response.json();
                    displayComparison(data.responses);
                } else {
                    addMessage('Error: Unable to get a response from the server.');
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('Error: Unable to connect to the server.');
            }
        } else if (Object.keys(selectedProviders).length === 0) {
            addMessage('Please select at least one provider and model.');
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
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        chatContainer.innerHTML = '';
        comparisonContainer.innerHTML = '';
        comparisonContainer.classList.add('hidden');
        addMessage('Conversation history cleared.');
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    clearHistoryBtn.addEventListener('click', clearHistory);
});
