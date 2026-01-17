/**
 * Invisible Feedback - Chat Application
 *
 * Handles:
 * - Session management
 * - Message sending/receiving
 * - Real-time metrics display
 * - Persona simulation
 */

// =============================================================================
// STATE
// =============================================================================

const state = {
    sessionId: null,
    personaId: null,
    status: 'idle', // idle, active, completed, exited
    messageStartTime: null,
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const elements = {
    personaSelect: document.getElementById('persona-select'),
    startBtn: document.getElementById('start-btn'),
    simulateBtn: document.getElementById('simulate-btn'),
    messagesArea: document.getElementById('messages-area'),
    messageInput: document.getElementById('message-input'),
    sendBtn: document.getElementById('send-btn'),
    chatStatus: document.getElementById('chat-status'),
    sessionIdDisplay: document.getElementById('session-id-display'),

    // Metrics
    efiValue: document.getElementById('efi-value'),
    efiBar: document.getElementById('efi-bar'),
    efiStatus: document.getElementById('efi-status'),
    idsValue: document.getElementById('ids-value'),
    idsBar: document.getElementById('ids-bar'),
    idsStatus: document.getElementById('ids-status'),
    npsValue: document.getElementById('nps-value'),
    npsBucket: document.getElementById('nps-bucket'),
    npsConfidence: document.getElementById('nps-confidence'),

    // Extractions
    extractionsList: document.getElementById('extractions-list'),
};

// =============================================================================
// API CALLS
// =============================================================================

const API_BASE = '';

async function fetchPersonas() {
    try {
        const response = await fetch(`${API_BASE}/chat/personas`);
        const data = await response.json();
        return data.personas;
    } catch (error) {
        console.error('Failed to fetch personas:', error);
        return [];
    }
}

async function startChat(personaId) {
    const response = await fetch(`${API_BASE}/chat/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            survey_id: 'apparel_post_purchase_v1',
            user_id: `demo_${Date.now()}`,
            persona_id: personaId || null,
        }),
    });
    return response.json();
}

async function sendMessage(sessionId, text, latencyMs) {
    const response = await fetch(`${API_BASE}/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            text: text,
            latency_ms: latencyMs,
        }),
    });
    return response.json();
}

async function simulateUser(sessionId) {
    const response = await fetch(`${API_BASE}/chat/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
        }),
    });
    return response.json();
}

// =============================================================================
// UI UPDATES
// =============================================================================

function populatePersonaSelect(personas) {
    elements.personaSelect.innerHTML = '<option value="">No persona (manual input)</option>';
    personas.forEach(persona => {
        const option = document.createElement('option');
        option.value = persona.id;
        option.textContent = `${persona.name} - ${persona.traits}`;
        elements.personaSelect.appendChild(option);
    });
}

function addMessage(text, role, animate = true) {
    // Remove empty state if present
    const emptyChat = elements.messagesArea.querySelector('.empty-chat');
    if (emptyChat) {
        emptyChat.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        ${text}
        <span class="message-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;

    if (animate) {
        messageDiv.style.opacity = '0';
        elements.messagesArea.appendChild(messageDiv);
        requestAnimationFrame(() => {
            messageDiv.style.opacity = '1';
        });
    } else {
        elements.messagesArea.appendChild(messageDiv);
    }

    // Scroll to bottom
    elements.messagesArea.scrollTop = elements.messagesArea.scrollHeight;
}

function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    elements.messagesArea.appendChild(indicator);
    elements.messagesArea.scrollTop = elements.messagesArea.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

function updateMetrics(metrics) {
    if (!metrics) return;

    // EFI
    if (metrics.efi) {
        const efi = metrics.efi.current;
        elements.efiValue.textContent = efi.toFixed(2);
        elements.efiBar.style.width = `${efi * 100}%`;

        // Color based on threshold
        elements.efiBar.classList.remove('medium', 'high');
        if (efi >= 0.5) {
            elements.efiBar.classList.add('high');
        } else if (efi >= 0.3) {
            elements.efiBar.classList.add('medium');
        }

        // Status text
        const action = metrics.efi.action;
        const statusMap = {
            'continue_normal': 'Engaged',
            'simplify': 'Mild friction',
            'quick_reply_options': 'Moderate friction',
            'one_last_question': 'High friction',
            'exit_graceful': 'Disengaged',
        };
        elements.efiStatus.textContent = statusMap[action] || action;
    }

    // IDS
    if (metrics.ids) {
        const ids = metrics.ids.current;
        elements.idsValue.textContent = ids.toFixed(2);
        elements.idsBar.style.width = `${ids * 100}%`;

        elements.idsBar.classList.remove('high');
        if (ids >= 0.3) {
            elements.idsBar.classList.add('high');
        }

        elements.idsStatus.textContent = metrics.ids.should_probe ? 'Low depth - probing' : 'Good depth';
    }

    // NPS
    if (metrics.nps) {
        const pAdv = metrics.nps.p_advocacy;
        elements.npsValue.textContent = Math.round(pAdv * 100);

        elements.npsBucket.textContent = metrics.nps.bucket;
        elements.npsBucket.className = `nps-bucket ${metrics.nps.bucket}`;

        elements.npsConfidence.textContent = `${Math.round(metrics.nps.confidence * 100)}% confidence`;
    }
}

function updateExtractions(extractions, existingFields = []) {
    if (!extractions || extractions.length === 0) {
        if (existingFields.length === 0) {
            elements.extractionsList.innerHTML = '<p class="empty-state">No data extracted yet</p>';
        }
        return;
    }

    // Clear empty state
    const emptyState = elements.extractionsList.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    extractions.forEach(extraction => {
        // Check if already displayed
        if (document.getElementById(`extraction-${extraction.field_id}`)) {
            return;
        }

        const item = document.createElement('div');
        item.className = 'extraction-item';
        item.id = `extraction-${extraction.field_id}`;
        item.innerHTML = `
            <div class="extraction-field">${formatFieldName(extraction.field_id)}</div>
            <div class="extraction-value">${extraction.extracted_value}</div>
            <div class="extraction-quote">"${extraction.quote}"</div>
            <div class="extraction-confidence">${Math.round(extraction.confidence * 100)}% confidence</div>
        `;
        elements.extractionsList.appendChild(item);
    });
}

function formatFieldName(fieldId) {
    return fieldId
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function updateStatus(status) {
    state.status = status;

    const statusMap = {
        'idle': 'Start a conversation',
        'in_progress': 'Active',
        'completed': 'Completed',
        'exited': 'Ended',
    };

    elements.chatStatus.textContent = statusMap[status] || status;

    // Update button states
    const isActive = status === 'in_progress';
    elements.messageInput.disabled = !isActive;
    elements.sendBtn.disabled = !isActive;
    elements.simulateBtn.disabled = !isActive || !state.personaId;
    elements.startBtn.disabled = isActive;
}

function enableInput() {
    elements.messageInput.disabled = false;
    elements.sendBtn.disabled = false;
    elements.messageInput.focus();
    state.messageStartTime = Date.now();
}

function disableInput() {
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

async function handleStart() {
    const personaId = elements.personaSelect.value;
    state.personaId = personaId || null;

    try {
        elements.startBtn.disabled = true;
        elements.startBtn.textContent = 'Starting...';

        const result = await startChat(personaId);

        state.sessionId = result.session_id;
        elements.sessionIdDisplay.textContent = result.session_id;

        // Clear messages area
        elements.messagesArea.innerHTML = '';

        // Add opening message
        addMessage(result.opening_message, 'bot');

        // Update status
        updateStatus('in_progress');

        // Enable input
        enableInput();

        elements.startBtn.textContent = 'Start Conversation';

    } catch (error) {
        console.error('Failed to start chat:', error);
        alert('Failed to start conversation. Check console for details.');
        elements.startBtn.disabled = false;
        elements.startBtn.textContent = 'Start Conversation';
    }
}

async function handleSend() {
    const text = elements.messageInput.value.trim();
    if (!text || !state.sessionId) return;

    // Calculate latency
    const latencyMs = state.messageStartTime
        ? Date.now() - state.messageStartTime
        : 5000;

    // Clear input
    elements.messageInput.value = '';
    disableInput();

    // Add user message
    addMessage(text, 'user');

    // Show typing indicator
    addTypingIndicator();

    try {
        const result = await sendMessage(state.sessionId, text, latencyMs);

        // Remove typing indicator
        removeTypingIndicator();

        // Add bot response
        addMessage(result.bot_response, 'bot');

        // Update metrics
        updateMetrics(result.metrics);

        // Update extractions
        updateExtractions(result.extractions, result.fields_completed);

        // Update status
        updateStatus(result.status);

        // Re-enable input if conversation continues
        if (result.status === 'in_progress') {
            enableInput();
        }

    } catch (error) {
        console.error('Failed to send message:', error);
        removeTypingIndicator();
        addMessage('Sorry, something went wrong. Please try again.', 'bot');
        enableInput();
    }
}

async function handleSimulate() {
    if (!state.sessionId || !state.personaId) return;

    elements.simulateBtn.disabled = true;
    elements.simulateBtn.textContent = 'Simulating...';

    try {
        const result = await simulateUser(state.sessionId);

        // Set the simulated message in input and send it
        elements.messageInput.value = result.user_message;

        // Small delay to show the message before sending
        await new Promise(resolve => setTimeout(resolve, 300));

        await handleSend();

    } catch (error) {
        console.error('Failed to simulate user:', error);
        alert('Failed to simulate user response.');
    } finally {
        elements.simulateBtn.textContent = 'Simulate User Response';
        if (state.status === 'in_progress' && state.personaId) {
            elements.simulateBtn.disabled = false;
        }
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    // Load personas
    const personas = await fetchPersonas();
    populatePersonaSelect(personas);

    // Event listeners
    elements.startBtn.addEventListener('click', handleStart);
    elements.sendBtn.addEventListener('click', handleSend);
    elements.simulateBtn.addEventListener('click', handleSimulate);

    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    elements.personaSelect.addEventListener('change', (e) => {
        state.personaId = e.target.value || null;
        if (state.status === 'in_progress') {
            elements.simulateBtn.disabled = !state.personaId;
        }
    });

    console.log('Invisible Feedback initialized');
}

// Start the app
init();
