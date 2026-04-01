"""
Grounding Interviewer — drives HAIC convention sessions grounded in satellite imagery.

Implements a Socratic dialogue loop that:
1. Anchors the participant in the satellite observation (stimulus)
2. Elicits phenomenological descriptions: what they *see*, *feel*, *associate*
3. Probes for specificity and lived-experience grounding
4. Avoids hollow AI-framing reflections (GFS: phrase_echo_avoidance)
5. Gradually deepens toward the HAIC lived_experience_axis

Uses the Anthropic API (claude-sonnet) when ANTHROPIC_API_KEY is set,
otherwise falls back to a deterministic heuristic turn-generator.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import ConventionSession

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Lived-experience axes for grounding (from HAIC spec)
_AXES = [
    "spatial_perception",
    "environmental_affect",
    "temporal_orientation",
    "embodied_scale",
    "phenomenological_presence",
]

_SYSTEM_PROMPT = """\
You are a grounding interviewer for the HumanAI Convention protocol.
Your role is to guide a human participant through a structured observation
of satellite imagery, eliciting genuine, lived-experience descriptions.

RULES:
- Always anchor responses to the specific satellite image shown
- Ask one clear question per turn — never more than two sentences
- Probe for specificity: colours, textures, spatial relationships, emotional resonance
- Never hollow-reflect: do not echo the user's words back as affirmation
- Avoid AI-framing phrases like "as an AI" or "from the satellite's perspective"
- Maintain curiosity-first tone; treat participant as domain expert of their own perception
- After 6+ user turns, begin closing with a synthesis question about what surprised them
- Maximum depth before close: 10 user turns

GROUNDING AXES (rotate through):
- Spatial perception: distances, scales, patterns, arrangements
- Environmental affect: what emotions or moods the scene evokes
- Temporal orientation: signs of change, age, seasonality
- Embodied scale: how human-sized the features feel
- Phenomenological presence: imagining being *in* the scene

SATELLITE CONTEXT will be provided in the first system message.
"""

# Heuristic fallback turns when Anthropic API unavailable
_OPENING_QUESTIONS = [
    "What is the first thing that catches your eye in this satellite image?",
    "When you look at this image, what stands out most immediately to you?",
    "Describe what you see — start with the most prominent feature.",
]

_DEPTH_QUESTIONS = [
    "What does the texture of that area remind you of at human scale?",
    "If you were standing at the centre of this image, what direction would you walk, and why?",
    "What time of year or time of day does this scene suggest to you?",
    "What human activity do you imagine has shaped what you're seeing?",
    "If this place has a feeling, what would you call it?",
    "What surprises you most about the scale of what you can see?",
    "What would you expect to smell or hear if you were physically there?",
    "Which part of this image feels most *alive* to you, and which feels most static?",
]

_CLOSING_QUESTIONS = [
    "Looking at the whole image now — what's the one detail you'd want someone else to notice?",
    "What aspect of this view challenged your expectations of what this region looks like?",
    "If this satellite image were the only record of this moment, what would it fail to capture?",
]


class Interviewer:
    """Manages a single HAIC grounding interview session."""

    def __init__(self, session: "ConventionSession"):
        self.session = session
        self._use_anthropic = bool(ANTHROPIC_API_KEY)
        self._client = None

        if self._use_anthropic:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                logger.warning("anthropic package not installed — using heuristic interviewer")
                self._use_anthropic = False

    def respond(self, user_message: str) -> str:
        """Generate the next interviewer question in response to a user message."""
        user_turns = [t for t in self.session.interview_turns if t.get("role") == "user"]
        turn_count = len(user_turns)  # count before appending current

        if self._use_anthropic and self._client:
            return self._anthropic_respond(user_message, turn_count)
        else:
            return self._heuristic_respond(user_message, turn_count)

    def _build_system_message(self) -> str:
        """Build system message including stimulus context."""
        parts = [_SYSTEM_PROMPT]

        if self.session.stimulus:
            s = self.session.stimulus
            parts.append(f"\nSATELLITE CONTEXT:")
            parts.append(f"Position: lon={s.satellite_position[0]:.4f}, lat={s.satellite_position[1]:.4f}, alt={s.satellite_position[2]:.1f}km")
            parts.append(f"Simulation time: {s.simulation_timestamp}")
            parts.append(f"Observation context: {s.observation_context}")

            for img in s.images:
                meta = img.metadata
                if meta:
                    parts.append(f"\nImage type: {img.stimulus_type.value}")
                    if meta.image_available:
                        parts.append("Image: available and shown to participant")
                        if meta.cloud_cover is not None:
                            parts.append(f"Cloud cover: {meta.cloud_cover:.1f}%")
                        if meta.source:
                            parts.append(f"Source: {meta.source}")
                        if meta.spectral_bands:
                            parts.append(f"Bands rendered: {', '.join(meta.spectral_bands)}")
                    else:
                        parts.append("Image: not available for this location/time")
        else:
            parts.append("\nNo satellite stimulus loaded — conduct a general Earth observation interview.")

        return "\n".join(parts)

    def _anthropic_respond(self, user_message: str, turn_count: int) -> str:
        """Use Claude to generate the next grounding question."""
        messages: List[Dict[str, str]] = []

        # Replay existing conversation
        for turn in self.session.interview_turns:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=self._build_system_message(),
                messages=messages,
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning("Anthropic API call failed: %s — falling back to heuristic", e)
            return self._heuristic_respond(user_message, turn_count)

    def _heuristic_respond(self, user_message: str, turn_count: int) -> str:
        """Deterministic heuristic fallback when API unavailable."""
        if turn_count == 0:
            return random.choice(_OPENING_QUESTIONS)
        elif turn_count >= 8:
            return random.choice(_CLOSING_QUESTIONS)
        else:
            # Pick a depth question we haven't used yet
            used = {t["content"] for t in self.session.interview_turns if t.get("role") == "assistant"}
            unused = [q for q in _DEPTH_QUESTIONS if q not in used]
            if unused:
                return random.choice(unused)
            return random.choice(_CLOSING_QUESTIONS)

    def get_opening(self) -> str:
        """Generate the opening message for a new session."""
        if self.session.stimulus:
            context = self.session.stimulus.observation_context
            image_count = sum(
                1 for img in self.session.stimulus.images
                if img.metadata and img.metadata.image_available
            )
            if image_count > 0:
                intro = (
                    f"I've loaded a satellite observation from orbit. "
                    f"{context} "
                    f"There {'is' if image_count == 1 else 'are'} {image_count} "
                    f"image{'s' if image_count > 1 else ''} available for you to examine. "
                )
            else:
                intro = (
                    "I have satellite telemetry for this pass, though imagery is not available "
                    "for this specific location right now. "
                )
        else:
            intro = "We're going to conduct a grounding observation session. "

        return intro + random.choice(_OPENING_QUESTIONS)
