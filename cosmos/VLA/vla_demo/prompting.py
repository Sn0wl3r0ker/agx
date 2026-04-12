from __future__ import annotations

from .schemas import MissionSpec, StepSpec


def build_reasoner_prompt(mission: MissionSpec, step: StepSpec) -> str:
    landmarks = ", ".join(step.expected_landmarks) if step.expected_landmarks else "none"
    return (
        "You are verifying the current AMR mission step from a recent camera clip. "
        "Judge only the active step and return strict JSON. "
        f"Mission: {mission.mission_text}. "
        f"Current step_id: {step.step_id}. "
        f"Instruction: {step.instruction}. "
        f"Visual goal: {step.visual_goal}. "
        f"Expected landmarks: {landmarks}. "
        "Return exactly one JSON object with keys "
        "step_id, step_match, step_completed, observed_landmarks, confidence, reason. "
        "Do not include markdown fences or extra text."
    )
