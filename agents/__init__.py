"""
Multi-agent fake news detection system.

Layered architecture:
  Ingestion → Claim Extraction → Specialist Agents → Orchestrator → Verdict
"""
from agents.base import AgentResult, BaseAgent
from agents.bias_sentiment import BiasSentimentAgent
from agents.claim_extraction import ClaimExtractionAgent
from agents.fact_check import FactCheckAgent
from agents.feedback import FeedbackLoop
from agents.ingestion import IngestionAgent
from agents.input_profile import InputProfile, InputType, analyze_input
from agents.knowledge_base import KnowledgeBase
from agents.media_forensics import MediaForensicsAgent
from agents.ml_classifier import MLClassifierAgent
from agents.orchestrator import Orchestrator
from agents.review_queue import ReviewQueue
from agents.source_credibility import SourceCredibilityAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    "BiasSentimentAgent",
    "ClaimExtractionAgent",
    "FactCheckAgent",
    "FeedbackLoop",
    "IngestionAgent",
    "InputProfile",
    "InputType",
    "KnowledgeBase",
    "MLClassifierAgent",
    "MediaForensicsAgent",
    "Orchestrator",
    "ReviewQueue",
    "SourceCredibilityAgent",
    "analyze_input",
]
