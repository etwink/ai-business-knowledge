"""Bulk document pipeline: folder scanning, dependency-aware clustering, hierarchical summarization."""

from .folder_scanner import FolderScanner, ScannedDocuments
from .cluster_builder import ClusterBuilder, DocumentCluster
from .hierarchical_summarizer import HierarchicalSummarizer, ClusterSummary
from .conversational_agent import ConversationalAgent, AgentResponse, DocumentUpdate, GapItem
from .context_agent import ProcessContextAgent, ProcessContext

__all__ = [
    "FolderScanner",
    "ScannedDocuments",
    "ClusterBuilder",
    "DocumentCluster",
    "HierarchicalSummarizer",
    "ClusterSummary",
    "ConversationalAgent",
    "AgentResponse",
    "DocumentUpdate",
    "GapItem",
    "ProcessContextAgent",
    "ProcessContext",
]
