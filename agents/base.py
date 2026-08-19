"""
Base classes for the multi-agent system.

Every specialist agent inherits from BaseAgent and returns an AgentResult.
The orchestrator collects AgentResults and produces a final Verdict.
"""
from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("fakenews.agents")


class Label(str, Enum):
    FAKE = "fake"
    REAL = "real"
    UNCERTAIN = "uncertain"


@dataclass
class AgentResult:
    """Standardised output every agent must produce."""
    agent_name: str
    label: Label
    confidence: float  # 0.0 – 1.0
    reasoning: str  # human-readable explanation
    evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent": self.agent_name,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "raw_output": self.raw_output,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class BaseAgent(ABC):
    """All agents inherit from this and implement `run()`."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.config = kwargs

    def __call__(self, article: dict[str, Any]) -> AgentResult:
        t0 = time.time()
        try:
            result = self.run(article)
            result.elapsed_ms = (time.time() - t0) * 1000
            logger.info(f"[{self.name}] label={result.label.value}  "
                        f"conf={result.confidence:.3f}  {result.elapsed_ms:.0f}ms")
            return result
        except Exception as exc:
            logger.error(f"[{self.name}] failed: {exc}", exc_info=True)
            return AgentResult(
                agent_name=self.name,
                label=Label.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Agent failed: {exc}",
                elapsed_ms=(time.time() - t0) * 1000,
            )

    @abstractmethod
    def run(self, article: dict[str, Any]) -> AgentResult:
        """Process an article dict and return an AgentResult.

        The article dict always contains at least:
          - title: str
          - text: str
          - url: str (may be empty)
          - images: list[str] (URLs of embedded images)
          - metadata: dict (source domain, author, published date, etc.)
        """
        ...
