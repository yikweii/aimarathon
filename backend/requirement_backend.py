from fastapi import HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import re
import time
from openai import OpenAI
from general import (app, firebase)
from config import api_key

client = OpenAI(base_url="https://llm.chutes.ai/v1", api_key=api_key)

TEXT_MODEL            = "Qwen/Qwen3-32B-TEE"
VISION_MODEL          = "google/gemma-4-31B-turbo-TEE"
EXTRACT_EVERY_N_TURNS = 2   # extract more frequently so progress updates quickly

# Fields we track for progress — user intent only, no recommendation outputs.
# camera_count / camera_arrangement are floor-plan outputs passed to the
# optimisation model; they don't count toward conversation completion.
TRACKED_FIELDS = [
    "site_type", "location", "size",
    "coverage_areas", "resolution",
    "night_vision", "retention_days", "remote_viewing",
    "budget", "connectivity",
    "camera_count",
]  # 12 fields — 72% threshold = ~9 fields needed

# ============================================================================
# Data Models
# ============================================================================

class FloorPlanData(BaseModel):
    name: str
    type: str
    data: str  # base64 data URL

class RequirementRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict]] = []
    floor_plan: Optional[FloorPlanData] = None
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    cached_requirements: Optional[Dict] = None  # last known requirements from frontend

class GenerateSummaryRequest(BaseModel):
    conversation_history: List[Dict]
    project_id: Optional[str] = None
    cached_requirements: Optional[Dict] = None  # skip re-extraction if provided

class RequirementResponse(BaseModel):
    bot_response: str
    requirements: Optional[Dict] = None
    stage: str
    summary: Optional[str] = None
    optimization_summary: Optional[str] = None
    progress: int = 0
    inferred_fields: List[str] = []

class ProjectData(BaseModel):
    """Project creation/update."""
    name: str
    client_email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    site_type: Optional[str] = None
    budget: Optional[str] = None

# ============================================================================
# Prompts
# ============================================================================

VISION_SYSTEM_PROMPT = """You are an experienced CCTV security consultant reviewing a client's floor plan.

Analyse the image naturally and conversationally — like a professional walking through the space with the client.

Cover what you can see, in a natural flow:
- What rooms, zones, or areas are visible
- Where the entry and exit points are (doors, gates, windows, staircases)
- Which areas are highest priority for security coverage
- Any blind spots or tricky angles that are hard to cover
- A rough estimate of how many cameras would be needed and why, with specific placement locations based on the floor plan (e.g. "one in the top-left corner of the living room facing the main entrance")

After giving the estimate, ask the client whether they are happy with that suggested quantity or if they have a different number in mind.

You do NOT need to use bullet points or a fixed format. Write like a consultant giving a verbal walkthrough — clear, confident, and helpful. If the floor plan is unclear or partial, say so and work with what you can see.

After your analysis, ask ONE natural follow-up question — the single most important thing still missing to design a good system. Keep it conversational, not clinical.

Do NOT ask multiple questions. Do NOT use [QUESTION] tags or any special markers."""

TEXT_SYSTEM_PROMPT = """You are a CCTV requirements consultant. Your only job is to gather the client's security requirements accurately and hand off cleanly. You are NOT a salesperson, procurement agent, or product recommender.

OUTPUT RULES — CRITICAL:
- Your reply is sent DIRECTLY to the client. Never output internal instructions, decision logic, bullet-point rules, or conditional statements (e.g. "If the client says X, reply with Y"). Those are your private instructions — never echo them.
- Write only what the client should actually read.

ASKING QUESTIONS:
- Ask ONE question per reply — the single most critical missing detail.
- Keep replies to 1–3 short sentences before the question.
- Infer what you can from context. Never ask about something already clear.
- If the client says "not sure", suggest a sensible default and move on.
- Never use bullet points, headers, or list multiple questions.

WHAT TO COLLECT (any order, skip what is already clear):
- Property type and rough size
- Which specific areas need coverage (interior / exterior / both)
- Priority zones (most important area)
- Resolution preference and night vision needed
- Recording retention period
- Remote viewing needed
- Budget range in RM
- Location / state in Malaysia

STRICT LIMITS — never do any of these:
- Do not name specific camera brands, models, or products.
- Do not quote prices or cost estimates.
- Do not name specific shops, dealers, or malls.
- Do not offer to arrange, source, book, or reach out to anyone.
- Do not make any promise the system cannot fulfil.

CLOSING: Only use the closing message when ALL of the following have been collected:
(1) property type
(2) coverage areas
(3) budget range in RM
(4) location / state in Malaysia
(5) at least one feature preference — resolution, night vision, OR remote viewing
(6) confirmed camera quantity (how many cameras the client wants)
(7) connectivity preference — wired OR wireless
(8) recording retention period (how many days of footage to keep)

If any of these are still missing, ask for the missing one instead of closing. Priority order to ask: budget → location → features → camera quantity → connectivity → retention. When all are present, say exactly:
"Thank you — I've captured your requirements. A consultant will follow up with suitable recommendations for your budget and location."
Then stop asking questions.

STATUS FLAG — append this on a new line at the very end of EVERY reply, no exceptions:
- If you are still gathering (any of the 9 items above are missing): <status>gathering</status>
- If you have said the closing message above: <status>complete</status>
Do not explain the tag. Do not skip it."""

PARSE_PROMPT = """/no_think
Extract CCTV security requirements from the conversation into one JSON object.

SOURCE RULE: Extract ONLY what the USER stated, asked for, or clearly accepted.
Do not include features the assistant suggested unless the user explicitly confirmed them.

PROPERTY FIELDS (include if discussed):
  site_type    : "residential" | "commercial"
  location     : city or state in Malaysia
  size         : brief label (e.g. "3-bedroom", "30x40 ft single-storey")
  budget       : as stated (e.g. "RM 5,000" or "RM 3,000–5,000")
  budget_tier  : "low" (under RM 3k) | "mid-range" (RM 3k–8k) | "premium" (above RM 8k)
  timeline     : when they want it installed
  connectivity : "wireless" | "wired"

COVERAGE FIELDS (include if discussed):
  coverage_areas  : list of zones (e.g. ["hallway", "living room", "front entrance"])
  coverage_focus  : "interior-only" | "exterior-only" | "both"
  priority_zone   : the most emphasised single area

USER PREFERENCES (include if discussed):
  resolution     : "720p" | "1080p" | "4K"
  night_vision   : true | false
  remote_viewing : true | false
  retention_days : number
  camera_count   : confirmed number of cameras the user wants (integer)

FLOOR PLAN FIELDS (only when a floor plan image was shared in the conversation):
  has_floor_plan   : true
  layout_type      : e.g. "single-storey" | "2-storey" | "open-plan"
  identified_zones : list of rooms visible in the floor plan
  entry_points     : list of doors or exits identified
  blind_spots      : coverage gaps or tricky angles noted
  camera_arrangement : list of camera placement objects derived from the floor plan analysis.
                       Each object must include:
                         "id"       : integer, starting at 1
                         "zone"     : room or area name (e.g. "living room", "front entrance")
                         "position" : descriptive placement (e.g. "top-left corner facing the main door")
                         "purpose"  : what this camera monitors (e.g. "monitor entry and exit traffic")
                       Example:
                       [
                         {"id": 1, "zone": "living room", "position": "top-left corner", "purpose": "monitor main entrance"},
                         {"id": 2, "zone": "hallway",     "position": "ceiling mid-point facing bedroom wing", "purpose": "track movement between zones"}
                       ]

ALWAYS INCLUDE THESE TWO ARRAYS:
  features : enabled requirement tags
             e.g. ["wireless", "night_vision", "no_remote", "self_install",
                   "interior_only", "30day_retention", "high_res"]
  keywords : decision-making signals
             e.g. ["residential", "urban-KL", "mid-range", "budget-conscious",
                   "interior-only", "hallway-focus", "single-storey"]

Return ONLY valid JSON. Omit any field not discussed. No nulls."""

SUMMARY_PROMPT = """You are a CCTV Technical Sales Consultant writing a Requirements Brief used to match the client with the right products and packages.

PURPOSE: This summary will be used by a matching model to select suitable CCTV products. Include EVERY detail collected — nothing should be omitted or assumed.

RULES:
- Include every single field that was discussed, even if the value is a default or inferred.
- If a value was not specified, use a sensible default and mark it "(recommended default)".
- If the user said "tight" budget, flag it and recommend cost-effective options.
- Ensure all recommendations are compatible with each other (cameras, NVR, storage, cabling).
- Flag any compatibility issues, missing info, or constraints explicitly.
- Do NOT include a Next Steps section.

Use this format:

## CCTV Requirements Summary

### Client & Property
- Building Type:
- Location:
- Size / Layout:
- Site Description:

### Coverage Requirements
- Coverage Focus (interior / exterior / both):
- Coverage Areas (list every zone):
- Priority Zone:
- Entry & Exit Points:

### Camera Specifications
- Camera Count:
- Camera Type:
- Resolution:
- Night Vision:
- Field of View Requirements:

### Floor Plan Details
(Include even if partial — list every room, zone, entry/exit, blind spot identified)
- Identified Zones:
- Entry Points:
- Blind Spots:
- Camera Arrangement (list each camera with zone and position):

### System Specifications
- NVR / DVR Recommendation:
- Storage Required:
- Connectivity (wired / wireless):

### Recording & Access
- Retention Period:
- Remote Viewing:
- Mobile App:

### Budget & Timeline
- Budget:
- Budget Tier:

### Compatibility & Flags
Note every compatibility consideration, inferred value, missing field, or constraint the matching model should be aware of."""

OPTIMIZATION_PROMPT = """/no_think
You are preparing a plain-text description for a CCTV camera placement optimisation model.

PURPOSE: This paragraph will be used to optimise camera positions and coverage. Focus ONLY on physical placement — zones, positions, entry/exit points, blind spots, and what each camera should cover.

Given the extracted requirements JSON, write ONE paragraph (4–6 sentences) that describes:
- Each camera's position and the zone it covers
- All entry and exit points that must be monitored
- Any blind spots or coverage gaps that need to be addressed
- The total camera count and coverage focus (interior / exterior / both)

Do NOT mention budget, brand, resolution, retention, or any non-spatial detail.
Write in plain prose. No bullet points, no headers. Output only the paragraph."""

# ============================================================================
# Helpers
# ============================================================================

def _to_storage_summary(markdown: str) -> str:
    """Convert structured markdown summary to paragraph prose for Firebase.
    Strips headers, bullets, tables and joins each section into one paragraph."""
    # Remove any leftover Next Steps section
    markdown = re.sub(r'###?\s*Next Steps.*', '', markdown, flags=re.DOTALL | re.IGNORECASE)

    paragraphs = []
    current: list = []

    for raw in markdown.split('\n'):
        line = raw.strip()

        if not line:
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue

        if line.startswith('#'):          # section header → paragraph break
            if current:
                paragraphs.append(' '.join(current))
                current = []
            continue

        if re.match(r'^\|[-| :]+\|$', line):   # table separator row
            continue

        if line.startswith('|') and line.endswith('|'):   # table data row
            cells = [c.strip() for c in line.strip('|').split('|')]
            cells = [re.sub(r'\*\*(.*?)\*\*', r'\1', c) for c in cells if c]
            if cells:
                current.append(', '.join(cells) + '.')
            continue

        line = re.sub(r'^[-*]\s+', '', line)          # strip bullet
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)  # strip bold
        line = re.sub(r'\*(.*?)\*', r'\1', line)       # strip italic
        if line:
            current.append(line)

    if current:
        paragraphs.append(' '.join(current))

    return '\n\n'.join(p for p in paragraphs if p.strip())


def _build_messages(history: List[Dict], user_message: str,
                    floor_plan: Optional[FloorPlanData] = None,
                    system: str = TEXT_SYSTEM_PROMPT,
                    known_requirements: Optional[Dict] = None) -> List[Dict]:
    if known_requirements:
        # Summarise confirmed fields so the LLM doesn't re-ask for them.
        confirmed = {k: v for k, v in known_requirements.items()
                     if v is not None and not k.startswith("_") and k not in ("features", "keywords")}
        if confirmed:
            context_block = (
                "\n\nALREADY CONFIRMED — do NOT ask about these again:\n"
                + "\n".join(f"  {k}: {v}" for k, v in confirmed.items())
            )
            system = system + context_block
    messages = [{"role": "system", "content": system}]
    for msg in history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })
    if floor_plan:
        user_content: Any = [
            {"type": "image_url", "image_url": {"url": floor_plan.data}},
            {"type": "text", "text": user_message},
        ]
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})
    return messages


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (handles unclosed tags too)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


def _strip_leaked_instructions(text: str) -> str:
    """Remove lines where the model echoed internal conditional logic."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that look like leaked system-prompt instructions
        if re.match(r'^-\s+If (the client|they|user)', stripped, re.IGNORECASE):
            continue
        if re.match(r'^-\s+If .+(say|says|confirms?|respond)', stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned).strip()


def _complete(messages: List[Dict], model: str = TEXT_MODEL,
              temperature: float = 0.7, max_tokens: int = 1500,
              retries: int = 2) -> str:
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _strip_leaked_instructions(_strip_thinking(response.choices[0].message.content))
        except Exception as e:
            last_exc = e
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[_complete] attempt {attempt + 1} failed ({type(e).__name__}: {e}) — retrying in {wait}s")
                time.sleep(wait)
    raise last_exc


# ── NEW: Infer fields from conversation text before asking ────────────────────
def infer_missing(requirements: Dict, conversation_text: str) -> tuple[Dict, List[str]]:
    """
    Attempt to fill missing fields by pattern-matching the conversation.
    Returns (updated_requirements, list_of_inferred_field_names).
    """
    text = conversation_text.lower()
    inferred = []

    # site_type
    if not requirements.get("site_type"):
        if any(w in text for w in ["shop", "retail", "office", "factory", "warehouse", "restaurant", "clinic", "mall"]):
            requirements["site_type"] = "commercial"
            inferred.append("site_type")
        elif any(w in text for w in ["house", "home", "apartment", "condo", "bungalow", "terrace", "semi-d", "flat"]):
            requirements["site_type"] = "residential"
            inferred.append("site_type")

    # location
    if not requirements.get("location"):
        location_map = {
            "kl": "Kuala Lumpur", "kuala lumpur": "Kuala Lumpur",
            "penang": "Penang", "johor": "Johor Bahru", "jb": "Johor Bahru",
            "selangor": "Selangor", "pj": "Petaling Jaya", "petaling jaya": "Petaling Jaya",
            "melaka": "Melaka", "malacca": "Melaka", "ipoh": "Ipoh",
            "sabah": "Sabah", "sarawak": "Sarawak", "kota kinabalu": "Kota Kinabalu",
        }
        for keyword, mapped in location_map.items():
            if keyword in text:
                requirements["location"] = mapped
                inferred.append("location")
                break

    # budget
    if not requirements.get("budget"):
        if any(w in text for w in ["tight", "budget", "cheap", "affordable", "low cost"]):
            requirements["budget"] = "budget-conscious"
            inferred.append("budget")
        else:
            # Try to find "rm XXXX" patterns
            match = re.search(r'rm\s*(\d[\d,]*)', text)
            if match:
                requirements["budget"] = f"RM {match.group(1)}"
                inferred.append("budget")

    # night_vision — only infer if user explicitly mentioned it (not bot suggestions)
    if "night_vision" not in requirements:
        if any(w in text for w in ["night vision", "night-vision", "24/7 recording", "dark area"]):
            requirements["night_vision"] = True
            inferred.append("night_vision")

    # remote_viewing — only infer from explicit user phrases
    if "remote_viewing" not in requirements:
        if any(w in text for w in ["view from phone", "check from phone", "mobile access", "remote access", "view remotely"]):
            requirements["remote_viewing"] = True
            inferred.append("remote_viewing")
        elif any(w in text for w in ["no remote", "don't need remote", "not remote"]):
            requirements["remote_viewing"] = False
            inferred.append("remote_viewing")

    return requirements, inferred


# ── NEW: Progress tracking ────────────────────────────────────────────────────
def get_progress(requirements: Dict) -> int:
    """Return 0–100 completion percentage based on tracked fields."""
    if not requirements:
        return 0
    filled = sum(1 for f in TRACKED_FIELDS if requirements.get(f))
    return round(filled / len(TRACKED_FIELDS) * 100)
# ─────────────────────────────────────────────────────────────────────────────


def extract_requirements(history: List[Dict]) -> Dict:
    conversation_text = ""
    floor_plan_shared = False
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if isinstance(content, list):
            floor_plan_shared = True
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        conversation_text += f"{role}: {content}\n\n"

    if floor_plan_shared:
        conversation_text = "[A floor plan image was shared in this conversation]\n\n" + conversation_text

    json_str = ""
    try:
        json_str = _complete(
            [
                {"role": "system", "content": PARSE_PROMPT},
                {"role": "user", "content": conversation_text},
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        result = json.loads(json_str.strip())
        if floor_plan_shared:
            result["has_floor_plan"] = True
        return result
    except Exception as e:
        print(f"[extract_requirements] failed — {type(e).__name__}: {e}")
        return {}


def generate_summary(history: List[Dict], requirements: Dict) -> str:
    conversation_text = ""
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        conversation_text += f"{role}: {content}\n\n"

    prompt = (
        f"Conversation:\n{conversation_text}\n\n"
        f"Extracted requirements JSON:\n{json.dumps(requirements, indent=2)}\n\n"
        f"Now write the full requirements summary."
    )
    try:
        return _complete(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3000,
        )
    except Exception as e:
        print(f"[generate_summary] LLM call failed — {type(e).__name__}: {e}")
        return ""


def generate_optimization_summary(requirements: Dict) -> str:
    """Generate a one-paragraph plain-text description for the CCTV placement optimisation model."""
    try:
        result = _complete(
            [
                {"role": "system", "content": OPTIMIZATION_PROMPT},
                {"role": "user", "content": f"Requirements JSON:\n{json.dumps(requirements, indent=2)}"},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        return result.strip()
    except Exception as e:
        print(f"[generate_optimization_summary] failed — {type(e).__name__}: {e}")
        return ""


def identify_missing_fields(requirements: Dict) -> List[str]:
    missing = []
    if not requirements.get("site_type"):
        missing.append("Building type")
    if not requirements.get("location"):
        missing.append("Location")
    if not requirements.get("coverage_areas"):
        missing.append("Coverage areas")
    if not requirements.get("budget"):
        missing.append("Budget")
    has_feature = (
        requirements.get("resolution") or
        requirements.get("night_vision") is not None or
        requirements.get("remote_viewing") is not None
    )
    if not has_feature:
        missing.append("Feature preferences (resolution / night vision / remote viewing)")
    if not requirements.get("camera_count"):
        missing.append("Camera quantity")
    if not requirements.get("connectivity"):
        missing.append("Connectivity (wired / wireless)")
    return missing


# Fields that must ALL be explicitly present before a conversation can be "complete".
# Feature gate: at least one of resolution/night_vision/remote_viewing must be set.
REQUIRED_FIELDS = {
    "location", "budget", "coverage_areas", "site_type",
    "camera_count", "connectivity",        # added — map to closing conditions 6 & 7
    "retention_days",      # added — closing conditions 8 & 9
}
REQUIRED_FEATURES = {"resolution", "night_vision", "remote_viewing"}

def determine_stage(requirements: Dict, user_turns: int = 0, chat_status: str = "gathering") -> str:
    # chat_status is the <status> tag extracted from the TEXT_PROMPT response.
    # It is the authoritative signal — PARSE field counts are used only for
    # progress display, not for deciding when the conversation is complete.
    if chat_status == "complete":
        return "complete"

    if not requirements:
        return "gathering"

    missing_required = [f for f in REQUIRED_FIELDS if not requirements.get(f)]
    if missing_required:
        return "clarifying" if get_progress(requirements) >= 40 else "gathering"

    has_feature = any(requirements.get(f) is not None for f in REQUIRED_FEATURES)
    if not has_feature:
        return "clarifying"

    return "clarifying"


def _user_text(history: List[Dict]) -> str:
    """Flatten only user turns — used for pattern-matching inference so bot phrasing
    doesn't falsely trigger inferred fields like night_vision or remote_viewing."""
    text = ""
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        text += content + " "
    return text

# ============================================================================
# Endpoints: Chat
# ============================================================================

@app.post("/chat", response_model=RequirementResponse)
async def chat(request: RequirementRequest) -> RequirementResponse:
    has_image = request.floor_plan is not None
    model  = VISION_MODEL if has_image else TEXT_MODEL
    system = VISION_SYSTEM_PROMPT if has_image else TEXT_SYSTEM_PROMPT

    messages = _build_messages(
        request.conversation_history,
        request.message,
        request.floor_plan,
        system=system,
        known_requirements=request.cached_requirements,
    )

    try:
        raw_response = _complete(messages, model=model)
    except Exception as e:
        print(f"Error calling {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raw_response = raw_response or "I'm sorry, I couldn't generate a response. Could you please rephrase or try again?"

    # Extract the <status> tag the chat model appended, then strip it from the reply.
    status_match = re.search(r"<status>(.*?)</status>", raw_response, re.IGNORECASE)
    chat_status = status_match.group(1).strip().lower() if status_match else "gathering"
    bot_response = re.sub(r"\s*<status>.*?</status>", "", raw_response, flags=re.IGNORECASE).strip()

    user_turn_count = sum(1 for m in request.conversation_history if m.get("role") == "user") + 1
    full_history = request.conversation_history + [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": bot_response},
    ]

    requirements = extract_requirements(full_history)
    inferred_fields: List[str] = []
    if not requirements and request.cached_requirements:
        requirements = request.cached_requirements
    requirements, inferred_fields = infer_missing(requirements, _user_text(full_history))

    progress = get_progress(requirements)
    stage = determine_stage(requirements, user_turns=user_turn_count, chat_status=chat_status)

    print(f"[chat] turn={user_turn_count} stage={stage} progress={progress}% missing={[f for f in REQUIRED_FIELDS if not requirements.get(f)]}")

    summary = None
    optimization_summary = None
    if stage == "complete":
        summary = generate_summary(full_history, requirements)
        optimization_summary = generate_optimization_summary(requirements)

    if firebase and request.project_id and request.conversation_id:
        try:
            firebase.save_message(conversation_id=request.conversation_id, role="user", content=request.message)
            firebase.save_message(conversation_id=request.conversation_id, role="assistant", content=bot_response)
            if requirements:
                firebase.save_requirements(
                    project_id=request.project_id,
                    requirements=requirements,
                    stage=stage,
                    summary=_to_storage_summary(summary) if summary else None,
                    optimization_summary=optimization_summary,
                    progress=progress,
                    inferred_fields=inferred_fields,
                )
        except Exception as e:
            print(f"Warning: Could not save to Firebase: {e}")

    return RequirementResponse(
        bot_response=bot_response,
        requirements=requirements,
        stage=stage,
        summary=summary,
        optimization_summary=optimization_summary,
        progress=progress,
        inferred_fields=inferred_fields,
    )


@app.post("/extract-requirements")
async def extract_requirements_endpoint(request: RequirementRequest) -> Dict:
    history = request.conversation_history + [{"role": "user", "content": request.message}]
    try:
        reqs = extract_requirements(history)
        reqs, inferred = infer_missing(reqs, _user_text(history))
        return {
            **reqs,
            "_progress": get_progress(reqs),
            "_inferred_fields": inferred,
            "_missing_fields": identify_missing_fields(reqs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize-requirements")
async def summarize_requirements_endpoint(requirements: Dict) -> Dict:
    summary_lines = ["Based on our conversation, here's what I understand:\n"]

    if requirements.get("site_type"):
        summary_lines.append(f"• **Building**: {requirements['site_type']}")

    if requirements.get("location"):
        summary_lines.append(f"• **Location**: {requirements['location']}")

    if requirements.get("size"):
        summary_lines.append(f"• **Size**: {requirements['size']}")

    if requirements.get("coverage_areas"):
        areas = requirements["coverage_areas"]
        summary_lines.append(f"• **Coverage Areas**: {', '.join(areas) if isinstance(areas, list) else areas}")

    if requirements.get("camera_type"):
        summary_lines.append(f"• **Camera Type**: {requirements['camera_type']}")

    if requirements.get("resolution"):
        summary_lines.append(f"• **Resolution**: {requirements['resolution']}")

    if requirements.get("night_vision") is not None:
        summary_lines.append(f"• **Night Vision**: {'Yes' if requirements['night_vision'] else 'No'}")

    if requirements.get("retention_days"):
        summary_lines.append(f"• **Recording Retention**: {requirements['retention_days']} days")

    if requirements.get("remote_viewing") is not None:
        summary_lines.append(f"• **Remote Viewing**: {'Yes' if requirements['remote_viewing'] else 'No'}")

    if requirements.get("budget"):
        summary_lines.append(f"• **Budget**: {requirements['budget']}")

    if requirements.get("timeline"):
        summary_lines.append(f"• **Timeline**: {requirements['timeline']}")

    progress = get_progress(requirements)
    summary_lines.append(f"\n**Completion**: {progress}%")

    return {
        "summary": "\n".join(summary_lines),
        "progress": progress,
        "missing_fields": identify_missing_fields(requirements),
    }

@app.post("/generate-summary")
async def generate_summary_endpoint(request: GenerateSummaryRequest) -> Dict:
    """
    Separate, on-demand endpoint for generating the full LLM requirements brief.
    Called by the frontend after stage reaches 'complete' so the chat response
    stays fast and the summary loads independently.
    """
    if not request.conversation_history:
        raise HTTPException(status_code=400, detail="No conversation history provided")

    if request.cached_requirements:
        requirements = request.cached_requirements
        requirements, inferred = infer_missing(requirements, _user_text(request.conversation_history))
    else:
        requirements = extract_requirements(request.conversation_history)
        requirements, inferred = infer_missing(requirements, _user_text(request.conversation_history))

    print(f"[/generate-summary] Generating summary — {len(requirements)} fields, project={request.project_id}")

    summary = generate_summary(request.conversation_history, requirements)
    if not summary:
        raise HTTPException(status_code=500, detail="Summary generation failed — LLM returned empty response")

    optimization_summary = generate_optimization_summary(requirements)

    if firebase and request.project_id:
        try:
            existing = firebase.get_requirements(request.project_id)
            merged_reqs = existing.get("data", requirements) if existing else requirements
            firebase.save_requirements(
                project_id=request.project_id,
                requirements=merged_reqs,
                stage="complete",
                summary=_to_storage_summary(summary),
                optimization_summary=optimization_summary,
                progress=existing.get("progress", get_progress(requirements)) if existing else get_progress(requirements),
                inferred_fields=existing.get("inferred_fields", inferred) if existing else inferred,
            )
            print(f"[/generate-summary] Saved to Firebase — project {request.project_id}")
        except Exception as e:
            print(f"[/generate-summary] Firebase save failed — {type(e).__name__}: {e}")

    # Return full structured markdown to frontend for display
    return {"summary": summary, "optimization_summary": optimization_summary, "requirements": requirements}

# ============================================================================
# Endpoints: Projects (Firebase)
# ============================================================================

@app.post("/projects", response_model=Dict)
async def create_project(project: ProjectData) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    try:
        project_id = firebase.create_project(project.dict())
        return {"project_id": project_id, "message": "Project created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}")
async def get_project(project_id: str) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    project = firebase.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.get("/projects")
async def list_projects(limit: int = Query(50, le=100)) -> List[Dict]:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    return firebase.list_projects(limit)


@app.put("/projects/{project_id}")
async def update_project(project_id: str, updates: Dict) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    success = firebase.update_project(project_id, updates)
    if not success:
        raise HTTPException(status_code=500, detail="Could not update project")
    return {"message": "Project updated"}


@app.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    success = firebase.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=500, detail="Could not delete project")
    return {"message": "Project deleted"}

# ============================================================================
# Endpoints: Conversations
# ============================================================================

@app.post("/projects/{project_id}/conversations", response_model=Dict)
async def create_conversation(project_id: str) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    try:
        conversation_id = firebase.create_conversation(project_id)
        return {"conversation_id": conversation_id, "project_id": project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_id}/history")
async def get_project_history(project_id: str) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    history = firebase.get_project_history(project_id)
    if not history["conversations"]:
        raise HTTPException(status_code=404, detail="No conversations found")
    return history

# ============================================================================
# Endpoints: Requirements
# ============================================================================

@app.get("/projects/{project_id}/requirements")
async def get_requirements(project_id: str) -> Dict:
    if not firebase:
        raise HTTPException(status_code=500, detail="Firebase not initialized")
    requirements = firebase.get_requirements(project_id)
    if not requirements:
        raise HTTPException(status_code=404, detail="Requirements not found")
    return requirements

# ============================================================================
# Endpoints: Health
# ============================================================================

@app.get("/health")
async def health_check():
    firebase_status = firebase.health_check() if firebase else {"status": "not configured"}
    return {
        "status": "ok",
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "firebase": firebase_status,
    }

# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting CCTV Requirements Chatbot Backend...")
    print(f"Server:       http://localhost:8000")
    print(f"Text model:   {TEXT_MODEL}")
    print(f"Vision model: {VISION_MODEL}")
    if firebase:
        print("✓ Firebase configured")
    else:
        print("⚠ Firebase not configured (optional)")
    uvicorn.run(app, host="0.0.0.0", port=8000)