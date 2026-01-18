# Invisible Feedback

**Surveys that feel like DMs** — Conversational feedback collection with real-time engagement metrics.

## Overview

Invisible Feedback transforms traditional surveys into natural DM conversations. Brands define their survey schema (like SurveyMonkey), but users experience a casual chat that feels human. The system:

- **Extracts structured data** from natural language responses
- **Computes real-time metrics**: Engagement Friction Index, Insight Depth Score, Inferred NPS
- **Adapts conversation flow** based on user engagement
- **Preserves partial data** even when users disengage

## Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone the repository
cd GorillaSurvey

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running

```bash
# Start the server
python -m uvicorn api.main:app --reload --port 8000

# Open in browser
open http://localhost:8000
```

### Docker

```bash
# Build and run with Docker
docker-compose up --build
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Chat UI         │    │  Dashboard       │                   │
│  │  (Instagram DM)  │    │  (Analytics)     │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
└───────────┼───────────────────────┼─────────────────────────────┘
            │                       │
            ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  POST /chat/start   POST /chat/message   GET /analytics          │
└─────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Modules                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Orchestrator│  │  Metrics    │  │ Extraction  │              │
│  │ (State)     │  │  (EFI/IDS)  │  │ (Fields)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Metrics

### Engagement Friction Index (EFI)

Measures user disengagement on a 0-1 scale. Components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Latency | 0.15 | Response time (60s = max friction) |
| Brevity | 0.25 | Short responses indicate disengagement |
| Sentiment | 0.20 | Negative sentiment = friction |
| Deflection | 0.30 | Patterns like "idk", "whatever" |
| Fatigue | 0.10 | Increases with conversation length |

**Thresholds:**
- 0.0-0.3: Engaged → Continue normally
- 0.3-0.5: Mild friction → Simplify questions
- 0.5-0.7: Moderate → Offer quick-reply options
- 0.7-0.85: High → "One last question?"
- 0.85-1.0: Disengaged → Exit gracefully

### Insight Depth Score (IDS)

Measures response quality on a 0-1 scale. Components:

- **Attributes**: Product-specific mentions (fit, fabric, size)
- **Causal markers**: "because", "since", "after"
- **Specificity**: Numbers, comparisons
- **Vagueness penalty**: Generic words ("fine", "okay")

### Inferred NPS

Predicts Net Promoter Score from conversation signals:

- **Promoter (p > 0.7)**: "love", "amazing", "recommend"
- **Passive (0.4-0.7)**: Neutral signals
- **Detractor (p < 0.4)**: "terrible", "returning", "waste"

## API Endpoints

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/start` | POST | Start new conversation |
| `/chat/message` | POST | Send user message |
| `/chat/state/{id}` | GET | Get conversation state |
| `/chat/sessions` | GET | List all sessions |
| `/chat/simulate` | POST | Simulate user response |
| `/chat/personas` | GET | List available personas |

### Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analytics` | GET | Aggregate metrics |
| `/analytics/dashboard` | GET | Full dashboard data |
| `/analytics/export/sessions` | GET | CSV export |

## Personas (Demo)

10 pre-defined personas for testing:

1. **Emma** (Helpful Enthusiast) - Detailed, positive
2. **Jake** (Brief Positive) - Short, happy responses
3. **Sarah** (Detailed Critic) - Balanced feedback
4. **Chris** (Vague Neutral) - Non-committal
5. **Alex** (Busy Deflector) - Wants to exit
6. **Morgan** (Silent Slow) - Minimal words
7. **Kevin** (Hostile Complainer) - Negative, frustrated
8. **Priya** (Thoughtful Suggester) - Constructive ideas
9. **Zoe** (Emoji Communicator) - Heavy emoji use
10. **Jordan** (Question Asker) - Deflects with questions

## Project Structure

```
invisible-feedback/
├── api/
│   ├── main.py              # FastAPI application
│   └── routes/
│       ├── chat.py          # Chat endpoints
│       └── analytics.py     # Analytics endpoints
├── src/
│   ├── metrics_tools.py     # EFI, IDS, NPS formulas
│   ├── extraction_tools.py  # Field extraction
│   ├── orchestrator_tools.py# State management
│   ├── survey_questions.py  # Survey questions loader
│   ├── survey_questions.json# Survey questions (plain text)
│   ├── personas.py          # Demo personas
│   └── analytics_tools.py   # Aggregation
├── frontend/
│   ├── index.html           # Chat UI
│   ├── dashboard.html       # Analytics dashboard
│   ├── styles.css           # Instagram-inspired styling
│   └── app.js               # Frontend logic
├── configs/agents/          # Solace Agent Mesh configs
├── demo/
│   ├── demo_script.md       # Judge presentation
│   └── personas.json        # Persona definitions
└── requirements.txt
```

## Known Limitations

- **No persistent storage**: Sessions lost on restart
- **Single survey**: Hardcoded apparel survey
- **English only**: No multilingual support
- **No authentication**: Open access
- **Unvalidated metrics**: Accuracy not benchmarked

## Future Improvements

1. Dynamic survey builder
2. Multi-language support
3. Persistent storage (Redis/Postgres)
4. A/B testing against traditional surveys
5. Real messaging platform integration
6. Brand voice fine-tuning

## License

MIT
