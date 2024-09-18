document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const providerSelect = document.getElementById('provider-select');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    function addMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message', isUser ? 'user-message' : 'bot-message');
        messageDiv.textContent = content;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, true);
            userInput.value = '';

            const provider = providerSelect.value;
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message, provider }),
                });

                if (response.ok) {
                    const data = await response.json();
                    addMessage(data.response);
                } else {
                    addMessage('Error: Unable to get a response from the server.');
                }
            } catch (error) {
                console.error('Error:', error);
                addMessage('Error: Unable to connect to the server.');
            }
        }
    }

    async function clearHistory() {
        const provider = providerSelect.value;
        try {
            const response = await fetch('/clear_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ provider }),
            });

            if (response.ok) {
                chatContainer.innerHTML = '';
                addMessage('Conversation history cleared.');
            } else {
                addMessage('Error: Unable to clear conversation history.');
            }
        } catch (error) {
            console.error('Error:', error);
            addMessage('Error: Unable to connect to the server.');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    clearHistoryBtn.addEventListener('click', clearHistory);
});
