// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const historyBtn = document.getElementById('historyBtn');
const savedBtn = document.getElementById('savedBtn');
const themeBtn = document.getElementById('themeBtn');
const historyPanel = document.getElementById('historyPanel');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const venueCardTemplate = document.getElementById('venueCardTemplate');

// State
let chatHistory = JSON.parse(localStorage.getItem('chatHistory')) || [];
let savedVenues = JSON.parse(localStorage.getItem('savedVenues')) || [];
let isDarkTheme = localStorage.getItem('theme') === 'dark';
let welcomeMessage = document.querySelector('.welcome-message');

// Event Listeners
sendBtn.addEventListener('click', handleSendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSendMessage();
});
historyBtn.addEventListener('click', toggleHistoryPanel);
closeHistoryBtn.addEventListener('click', toggleHistoryPanel);
savedBtn.addEventListener('click', showSavedVenues);
themeBtn.addEventListener('click', toggleTheme);

// Initialize
loadChatHistory();
checkTheme();

// Functions
async function handleSendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Remove welcome message with fade effect
    if (welcomeMessage) {
        welcomeMessage.style.opacity = '0';
        welcomeMessage.style.transition = 'opacity 0.5s ease-out';
        setTimeout(() => {
            welcomeMessage.remove();
            welcomeMessage = null;
        }, 500);
    }

    // Add user message to chat
    addMessage(message, 'user');
    userInput.value = '';

    // Show loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'spinner';
    chatMessages.appendChild(loadingDiv);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                message,
                context: {
                    eventType: getEventType(message),
                    location: getLocation(message),
                    capacity: getCapacity(message),
                    budget: getBudget(message)
                }
            })
        });

        const data = await response.json();
        
        // Remove loading indicator
        loadingDiv.remove();

        // Handle the response
        if (data.error) {
            addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        } else {
            addMessage(data.response, 'bot');
            
            // If venues are included in the response, display them
            if (data.venues && data.venues.length > 0) {
                displayVenues(data.venues);
            } else {
                addMessage('I couldn\'t find any venues matching your criteria. Try being more specific about the type of venue you\'re looking for.', 'bot');
            }
        }

        // Save to history
        saveChatHistory(message, data.response, data.venues);
    } catch (error) {
        console.error('Error:', error);
        loadingDiv.remove();
        addMessage('Sorry, I encountered an error. Please try again.', 'bot');
    }

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function getEventType(message) {
    const eventTypes = {
        'business|meeting|conference|corporate|office|work': 'business',
        'sports|game|tournament|match|stadium|arena|gym|field|court': 'sports',
        'wedding|reception|marriage|ceremony|bridal': 'wedding',
        'party|celebration|social|event|gathering|get-together': 'social',
        'graduation|ceremony|commencement|convocation': 'graduation',
        'exhibition|gallery|art|show|museum|display': 'exhibition',
        'restaurant|cafe|dining|food|eat|dinner|lunch|breakfast': 'dining',
        'hotel|resort|accommodation|stay|lodging': 'accommodation',
        'theater|cinema|movie|play|performance|show': 'entertainment'
    };

    for (const [keywords, type] of Object.entries(eventTypes)) {
        if (new RegExp(keywords, 'i').test(message)) {
            return type;
        }
    }
    return 'general';
}

function getLocation(message) {
    // Extract location using regex
    const locationMatch = message.match(/\b(in|at|near|around|close to|within)\s+([^,.!?]+)/i);
    return locationMatch ? locationMatch[2].trim() : null;
}

function getCapacity(message) {
    // Extract capacity using regex
    const capacityMatch = message.match(/\b(\d+)\s*(people|guests|attendees|capacity|persons|individuals)\b/i);
    return capacityMatch ? parseInt(capacityMatch[1]) : null;
}

function getBudget(message) {
    // Extract budget using regex
    const budgetMatch = message.match(/\b(\$|₹|€|£)?\s*(\d+)\s*(k|thousand|K)?\b/i);
    if (budgetMatch) {
        let amount = parseInt(budgetMatch[2]);
        if (budgetMatch[3]) amount *= 1000;
        return amount;
    }
    return null;
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
}

function displayVenues(venues) {
    const venuesGrid = document.createElement('div');
    venuesGrid.className = 'venue-grid';
    
    venues.forEach(venue => {
        const template = document.getElementById('venueCardTemplate');
        const card = template.content.cloneNode(true);
        
        const venueCard = card.querySelector('.venue-card');
        const image = card.querySelector('.venue-image');
        const name = card.querySelector('.venue-name');
        const type = card.querySelector('.venue-type');
        const details = card.querySelector('.venue-details');
        const address = card.querySelector('.address');
        const directionsLink = card.querySelector('.directions-link');
        const saveButton = card.querySelector('.save-venue');
        
        image.src = venue.image || '/static/placeholder.jpg';
        name.textContent = venue.name;
        type.textContent = venue.type;
        
        // Create venue details
        const detailsHTML = [];
        if (venue.rating) detailsHTML.push(`<span><i class="fas fa-star"></i> ${venue.rating}</span>`);
        if (venue.price) detailsHTML.push(`<span><i class="fas fa-dollar-sign"></i> ${venue.price}</span>`);
        if (venue.capacity) detailsHTML.push(`<span><i class="fas fa-users"></i> ${venue.capacity}</span>`);
        details.innerHTML = detailsHTML.join('');
        
        address.textContent = venue.address;
        
        const mapsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(venue.name + ' ' + venue.address)}`;
        directionsLink.href = mapsUrl;
        
        const isVenueSaved = savedVenues.some(v => v.name === venue.name);
        saveButton.innerHTML = `<i class="fa${isVenueSaved ? 's' : 'r'} fa-bookmark"></i>`;
        saveButton.onclick = () => toggleSaveVenue(venue, saveButton);
        
        venuesGrid.appendChild(card);
    });
    
    chatMessages.appendChild(venuesGrid);
}

function saveVenue(venue) {
    if (!savedVenues.some(v => v.name === venue.name)) {
        savedVenues.push(venue);
        localStorage.setItem('savedVenues', JSON.stringify(savedVenues));
        showNotification('Venue saved!');
    }
}

function showSavedVenues() {
    chatMessages.innerHTML = '';
    
    if (savedVenues.length === 0) {
        addMessage('No saved venues yet.', 'bot');
        return;
    }
    
    addMessage('Here are your saved venues:', 'bot');
    displayVenues(savedVenues);
}

function saveChatHistory(userMessage, botResponse, venues) {
    const historyItem = {
        timestamp: new Date().toISOString(),
        userMessage,
        botResponse,
        venues
    };
    chatHistory.unshift(historyItem);
    if (chatHistory.length > 10) chatHistory.pop();
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    updateHistoryPanel();
}

function loadChatHistory() {
    const historyContent = document.getElementById('historyContent');
    historyContent.innerHTML = '';
    
    chatHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-timestamp">${new Date(item.timestamp).toLocaleString()}</div>
            <div class="history-user-message">${item.userMessage}</div>
            <div class="history-bot-response">${item.botResponse}</div>
        `;
        historyContent.appendChild(historyItem);
    });
}

function toggleHistoryPanel() {
    historyPanel.classList.toggle('active');
}

function toggleTheme() {
    const isDark = !isDarkTheme;
    isDarkTheme = isDark;
    document.body.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    themeBtn.innerHTML = isDark ? '<i class="fas fa-moon"></i> Theme' : '<i class="fas fa-sun"></i> Theme';
}

function checkTheme() {
    document.body.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
    themeBtn.innerHTML = isDarkTheme ? '<i class="fas fa-moon"></i> Theme' : '<i class="fas fa-sun"></i> Theme';
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function toggleSaveVenue(venue, button) {
    const index = savedVenues.findIndex(v => v.name === venue.name);
    if (index === -1) {
        savedVenues.push(venue);
        button.innerHTML = '<i class="fas fa-bookmark"></i>';
        button.classList.add('saved');
    } else {
        savedVenues.splice(index, 1);
        button.innerHTML = '<i class="far fa-bookmark"></i>';
        button.classList.remove('saved');
    }
    localStorage.setItem('savedVenues', JSON.stringify(savedVenues));
}

function updateHistoryPanel() {
    const historyContent = document.getElementById('historyContent');
    historyContent.innerHTML = '';
    
    chatHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-timestamp">${new Date(item.timestamp).toLocaleString()}</div>
            <div class="history-user-message">${item.userMessage}</div>
            <div class="history-bot-response">${item.botResponse}</div>
        `;
        historyContent.appendChild(historyItem);
    });
} 
