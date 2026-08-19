"""
Multi-agent fake news detection system.

Layered architecture:
  Ingestion → Claim Extraction → Specialist Agents → Orchestrator → Verdict
"""
from agents.base import BaseAgent, AgentResult
from agents.ingestion import IngestionAgent
from agents.claim_extraction import ClaimExtractionAgent
from agents.ml_classifier import MLClassifierAgent
from agents.fact_check import FactCheckAgent
from agents.source_credibility import SourceCredibilityAgent
from agents.media_forensics import MediaForensicsAgent
from agents.bias_sentiment import BiasSentimentAgent
from agents.orchestrator import Orchestrator
from agents.knowledge_base import KnowledgeBase
from agents.review_queue import ReviewQueue
from agents.feedback import FeedbackLoop

__all__ = [
    "BaseAgent", "AgentResult",
    "IngestionAgent", "ClaimExtractionAgent", "MLClassifierAgent",
    "FactCheckAgent", "SourceCredibilityAgent", "MediaForensicsAgent",
    "BiasSentimentAgent", "Orchestrator", "KnowledgeBase",
    "ReviewQueue", "FeedbackLoop",
]
