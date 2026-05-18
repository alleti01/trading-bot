"""ModelReviewAgent: calibration / drift commentary on the active model.

Reads model metadata (latest from ``models/model_registry``) plus any
prediction rows recorded today and emits a short calibration comment
with optional drift warnings and a *purely advisory* retrain
recommendation. Promotion of any new model still requires the
deterministic walk-forward workflow.
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import ModelReviewOutput


class ModelReviewAgent(BaseAgent):
    name: ClassVar[str] = "model_review"
    schema_class = ModelReviewOutput

    system_prompt: ClassVar[str] = (
        "You are a model-monitoring assistant. "
        "Given the active model's training metrics and today's prediction "
        "distribution, comment on calibration and surface drift warnings. "
        "Set retrain_recommended=true only when there is clear evidence: "
        "e.g. >2x calibration error vs training, or sustained probability "
        "drift relative to recorded validation distribution. Be conservative; "
        "false alarms cause unnecessary retrains. "
        "You cannot promote, demote, or modify any model. "
        "Return JSON matching ModelReviewOutput exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        meta = context.model_metadata or {}
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "instrument": context.instrument,
                    "model_name": meta.get("name", "none"),
                    "model_version": meta.get("version", "none"),
                    "training_metrics": meta.get("metrics", {}),
                    "calibration_table": meta.get("calibration_table", []),
                    "feature_columns": meta.get("feature_columns", []),
                    "today_predictions": meta.get("today_predictions", []),
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: model_name (string), "
              "model_version (string), calibration_comment (string), "
              "drift_warnings (string list), retrain_recommended (bool), "
              "reason (string)."
        )
