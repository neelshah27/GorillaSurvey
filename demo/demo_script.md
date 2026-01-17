# Invisible Feedback - Judge Demo Script

**Duration**: 3 minutes
**Demo URL**: http://localhost:8000

---

## SLIDE 1: The Problem (30 seconds)

> "70% of survey respondents abandon before completion. Why? Because surveys feel like work—numbered questions, radio buttons, progress bars. Users disengage because the format screams 'give us data.'"

> "What if we could get the same structured data through a conversation that feels like texting a friend?"

---

## SLIDE 2: The Solution (30 seconds)

> "Introducing Invisible Feedback: surveys that disappear as an interface."

> "Brands define their survey schema—same fields as SurveyMonkey. But users experience a casual DM conversation. No 'on a scale of 1-10,' no checkboxes. Just natural chat."

> "Behind the scenes, we extract structured responses, compute real-time engagement metrics, and infer NPS without ever asking directly."

---

## LIVE DEMO (90 seconds)

### Start the conversation

1. Open http://localhost:8000
2. Select persona: **"Emma - Helpful Enthusiast"**
3. Click **"Start Conversation"**

> "Here's a user who just bought from ThreadCraft, our demo apparel brand. Watch how the conversation flows."

### First exchange

4. The bot sends: *"Hey! Thanks for shopping with ThreadCraft 🧵 Mind if I ask a few quick questions about your recent order?"*
5. Click **"Simulate User Response"** (or type): *"loved it! fabric is amazing, fits perfect"*

> "Notice the metrics panel updating in real-time."

**Point to sidebar:**
- "EFI at 0.15—highly engaged"
- "IDS at 0.72—specific, actionable insight"
- "NPS inferred as Promoter based on sentiment signals"

**Point to extractions:**
- "The system extracted 'overall_satisfaction: loved' and 'fabric_quality: excellent' with confidence scores"

### Second exchange

6. Bot asks about fit naturally
7. Simulate response: *"Perfect fit! Usually I struggle with sizing but this was spot on"*

> "No numbered questions, no rating scales. Just a natural conversation—but we're capturing the same data."

### Show graceful handling

8. If time permits, switch to **"busy_deflector"** persona
9. Show how EFI increases and system offers quick-reply options or exits gracefully

> "When users disengage, the system detects it and adapts. High friction? Offer quick choices or exit gracefully. We preserve partial data rather than losing everything."

---

## DASHBOARD (30 seconds)

1. Click **"Open Dashboard"** (or go to /dashboard)

> "Here's what brands see: aggregate NPS, completion rates, average engagement scores."

**Point to key metrics:**
- "Real-time completion rate"
- "Top feedback themes—extracted automatically"
- "Live session monitoring"

> "Same data quality as traditional surveys, 3x the completion rate potential."

---

## CLOSE (20 seconds)

> "We built this in 6 hours using a multi-agent architecture:"
- "Extraction Agent maps natural language to schema fields"
- "Metrics Agent computes engagement friction and insight depth"
- "Writer Agent generates on-brand responses"
- "All coordinated through Solace Agent Mesh"

> "The future of feedback isn't forms—it's conversations. Questions?"

---

## BACKUP TALKING POINTS

**If asked about metrics:**
- EFI combines latency, brevity, sentiment, deflection patterns, and conversation fatigue
- IDS measures product attribute mentions, causal language, specificity
- NPS uses Bayesian updates based on positive/negative signal detection

**If asked about accuracy:**
- Rule-based extraction achieves ~85% precision on demo data
- LLM fallback for ambiguous cases
- Confidence scores let brands review uncertain extractions

**If asked about privacy:**
- Transparent opening—users know they're giving feedback
- No PII extraction
- All processing happens per-session, no long-term user tracking

**If asked about scaling:**
- Event-driven architecture (Solace) handles concurrent sessions
- Stateless API design
- In-memory for demo, Redis/Postgres for production
