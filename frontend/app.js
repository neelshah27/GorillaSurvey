/**
 * Invisible Feedback - Chat Application
 *
 * Handles:
 * - Session management
 * - Message sending/receiving
 * - Real-time metrics display
 * - Session management
 */

// =============================================================================
// STATE
// =============================================================================

const state = {
    sessionId: null,
    status: 'idle', // idle, active, completed, exited
    messageStartTime: null,
    questions: {},
    askedQuestions: new Set(),
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const elements = {
    startBtn: document.getElementById('start-btn'),
    endBtn: document.getElementById('end-btn'),
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

    // Progress
    progressCount: document.getElementById('progress-count'),
    progressFill: document.getElementById('progress-fill'),
    progressNote: document.getElementById('progress-note'),
    questionList: document.getElementById('question-list'),
};

// =============================================================================
// API CALLS
// =============================================================================

const API_BASE = '';

async function startChat() {
    const response = await fetch(`${API_BASE}/chat/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            survey_id: 'apparel_post_purchase_v1',
            user_id: `demo_${Date.now()}`,
            persona_id: null,
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

async function endChat(sessionId) {
    const response = await fetch(`${API_BASE}/chat/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            status: 'completed',
        }),
    });
    return response.json();
}

async function fetchSessionState(sessionId) {
    const response = await fetch(`${API_BASE}/chat/state/${sessionId}`);
    return response.json();
}

// =============================================================================
// UI UPDATES
// =============================================================================

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
    elements.startBtn.disabled = isActive;
    elements.endBtn.disabled = !isActive;
}

function renderProgress(sessionState) {
    if (!sessionState || !sessionState.questions) {
        elements.progressCount.textContent = '0/0';
        elements.progressFill.style.width = '0%';
        elements.progressNote.textContent = 'Waiting to start';
        elements.questionList.innerHTML = '<p class="empty-state">No questions loaded</p>';
        return;
    }

    const questions = sessionState.questions;
    const entries = Object.entries(questions);
    const total = entries.length;
    const answered = new Set(Object.keys(sessionState.extracted_fields || {}));

    if (sessionState.next_field_target) {
        state.askedQuestions.add(sessionState.next_field_target);
    }

    const answeredCount = answered.size;
    const percent = total > 0 ? Math.round((answeredCount / total) * 100) : 0;

    elements.progressCount.textContent = `${answeredCount}/${total}`;
    elements.progressFill.style.width = `${percent}%`;
    elements.progressNote.textContent = total > 0
        ? `${percent}% complete`
        : 'Waiting to start';

    if (total === 0) {
        elements.questionList.innerHTML = '<p class="empty-state">No questions loaded</p>';
        return;
    }

    elements.questionList.innerHTML = '';
    entries.forEach(([id, text]) => {
        const item = document.createElement('div');
        const isAnswered = answered.has(id);
        const isCurrent = sessionState.next_field_target === id && !isAnswered;
        const isAsked = state.askedQuestions.has(id) && !isAnswered && !isCurrent;

        item.className = 'question-item';
        if (isAnswered) {
            item.classList.add('answered');
        } else if (isCurrent) {
            item.classList.add('current');
        } else if (isAsked) {
            item.classList.add('asked');
        }

        item.innerHTML = `
            <span class="status-dot"></span>
            <span class="question-text">${text}</span>
        `;

        elements.questionList.appendChild(item);
    });
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
    try {
        elements.startBtn.disabled = true;
        elements.startBtn.textContent = 'Starting...';

        state.askedQuestions = new Set();
        state.questions = {};
        renderProgress(null);

        const result = await startChat();

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
        elements.endBtn.disabled = false;

        try {
            const sessionState = await fetchSessionState(state.sessionId);
            state.questions = sessionState.questions || {};
            renderProgress(sessionState);
        } catch (error) {
            console.error('Failed to load session state:', error);
        }

        elements.startBtn.textContent = 'Start Conversation';

    } catch (error) {
        console.error('Failed to start chat:', error);
        alert('Failed to start conversation. Check console for details.');
        elements.startBtn.disabled = false;
        elements.startBtn.textContent = 'Start Conversation';
    }
}

async function handleEnd() {
    if (!state.sessionId) return;

    elements.endBtn.disabled = true;
    try {
        const result = await endChat(state.sessionId);
        if (result.closing_message) {
            addMessage(result.closing_message, 'bot');
        }
        updateStatus(result.status || 'completed');
        disableInput();
    } catch (error) {
        console.error('Failed to end chat:', error);
        elements.endBtn.disabled = false;
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

        // Update status
        updateStatus(result.status);

        try {
            const sessionState = await fetchSessionState(state.sessionId);
            state.questions = sessionState.questions || {};
            renderProgress(sessionState);
        } catch (error) {
            console.error('Failed to refresh session state:', error);
        }

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

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    // Event listeners
    elements.startBtn.addEventListener('click', handleStart);
    elements.endBtn.addEventListener('click', handleEnd);
    elements.sendBtn.addEventListener('click', handleSend);

    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    console.log('Invisible Feedback initialized');
}

// Start the app
init();
