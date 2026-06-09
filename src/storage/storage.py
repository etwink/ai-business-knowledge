"""Storage and persistence layer for analysis results."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from src.analyzers import ProcessDocument, GapAnalysis, AnalysisResult


class AnalysisStorage:
    """Manages saving and loading analysis results."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize storage manager.
        
        Args:
            storage_dir: Directory to store analysis files. Defaults to './analysis_sessions'
        """
        self.storage_dir = Path(storage_dir or "./analysis_sessions")
        self.storage_dir.mkdir(exist_ok=True)

    def create_session(self, session_name: str) -> Path:
        """
        Create a new analysis session directory.
        
        Args:
            session_name: Name of the session
            
        Returns:
            Path to the session directory
        """
        session_dir = self.storage_dir / session_name
        session_dir.mkdir(exist_ok=True)
        return session_dir

    def save_analyses(
        self,
        session_name: str,
        analyses: List[AnalysisResult]
    ) -> Path:
        """Save individual document analyses."""
        session_dir = self.create_session(session_name)
        
        analyses_data = []
        for analysis in analyses:
            analyses_data.append({
                "document_name": analysis.document_name,
                "summary": analysis.summary,
                "key_processes": analysis.key_processes,
                "systems_mentioned": analysis.systems_mentioned,
                "technical_details": analysis.technical_details
            })
        
        file_path = session_dir / "analyses.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(analyses_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def save_process_document(
        self,
        session_name: str,
        process_doc: ProcessDocument,
        version: str = "v1"
    ) -> Path:
        """Save generated process document."""
        session_dir = self.create_session(session_name)
        
        doc_data = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "overview": process_doc.overview,
            "integrated_processes": process_doc.integrated_processes,
            "dependencies": process_doc.dependencies,
            "data_flow": process_doc.data_flow,
            "decision_points": process_doc.decision_points,
            "systems_and_components": process_doc.systems_and_components,
            "appendix": process_doc.appendix,
            "process_flow_diagram": process_doc.process_flow_diagram,
            "process_flow_ascii": process_doc.process_flow_ascii,
            "citations": process_doc.citations,
        }
        
        file_path = session_dir / f"process_document_{version}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def save_gap_analysis(
        self,
        session_name: str,
        gap_analysis: GapAnalysis,
        version: str = "v1"
    ) -> Path:
        """Save gap analysis results."""
        session_dir = self.create_session(session_name)
        
        gaps_data = {
            "version": version,
            "format": "v2",
            "created_at": datetime.now().isoformat(),
            "gaps_by_category": gap_analysis.gaps_by_category,
            "edge_cases": gap_analysis.edge_cases,
            "resource_gaps": gap_analysis.resource_gaps,
            "ranked_gaps": gap_analysis.ranked_gaps,
        }
        
        file_path = session_dir / f"gap_analysis_{version}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(gaps_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def save_questions(
        self,
        session_name: str,
        questions: List[Dict[str, str]],
        version: str = "v1"
    ) -> Path:
        """Save clarification questions."""
        session_dir = self.create_session(session_name)
        
        questions_data = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "questions": questions
        }
        
        file_path = session_dir / f"questions_{version}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def save_user_answers(
        self,
        session_name: str,
        answers: Dict[int, str]
    ) -> Path:
        """Save user's answers to clarification questions."""
        session_dir = self.create_session(session_name)
        
        answers_data = {
            "created_at": datetime.now().isoformat(),
            "answers": answers
        }
        
        file_path = session_dir / "user_answers.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(answers_data, f, indent=2, ensure_ascii=False)
        
        return file_path

    def load_session(self, session_name: str) -> Dict[str, Any]:
        """Load all data from a session."""
        session_dir = self.storage_dir / session_name
        
        if not session_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_name}")
        
        session_data = {}
        
        # Load analyses
        analyses_file = session_dir / "analyses.json"
        if analyses_file.exists():
            with open(analyses_file, 'r', encoding='utf-8') as f:
                session_data['analyses'] = json.load(f)
        
        # Load latest process document
        process_doc_files = sorted(session_dir.glob("process_document_*.json"))
        if process_doc_files:
            with open(process_doc_files[-1], 'r', encoding='utf-8') as f:
                session_data['process_document'] = json.load(f)
        
        # Load latest gap analysis
        gap_files = sorted(session_dir.glob("gap_analysis_*.json"))
        if gap_files:
            with open(gap_files[-1], 'r', encoding='utf-8') as f:
                session_data['gap_analysis'] = json.load(f)
        
        # Load chat messages
        chat_file = session_dir / "chat_messages.json"
        if chat_file.exists():
            with open(chat_file, 'r', encoding='utf-8') as f:
                session_data['chat_messages'] = json.load(f)

        # Load document updates
        doc_updates = self.load_document_updates(session_name)
        if doc_updates:
            session_data['document_updates'] = doc_updates

        # Load session settings (audience, process context)
        settings_file = session_dir / "session_settings.json"
        if settings_file.exists():
            with open(settings_file, 'r', encoding='utf-8') as f:
                session_data['settings'] = json.load(f)

        return session_data

    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        if not self.storage_dir.exists():
            return []
        
        return sorted([d.name for d in self.storage_dir.iterdir() if d.is_dir()])

    def save_document_update(
        self,
        session_name: str,
        document_name: str,
        version: int,
        content: str,
        gap_addressed: str = "",
    ) -> Path:
        """Save an updated version of a source document produced by the chat agent."""
        session_dir = self.create_session(session_name)
        updates_dir = session_dir / "document_updates"
        updates_dir.mkdir(exist_ok=True)

        import re
        safe_name = re.sub(r'[^\w\s-]', '_', document_name)
        file_path = updates_dir / f"{safe_name}_v{version}.json"

        data = {
            "document_name": document_name,
            "version": version,
            "content": content,
            "gap_addressed": gap_addressed,
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return file_path

    def save_settings(self, session_name: str, settings: dict) -> Path:
        """Save session-level settings: audience selection and process context."""
        session_dir = self.create_session(session_name)
        file_path = session_dir / "session_settings.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return file_path

    def save_chat_messages(self, session_name: str, messages: List[Dict]) -> Path:
        """Save chat message history."""
        session_dir = self.create_session(session_name)
        file_path = session_dir / "chat_messages.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        return file_path

    def load_document_updates(self, session_name: str) -> Dict[str, List[Dict]]:
        """Load all saved document update versions for a session."""
        session_dir = self.storage_dir / session_name
        updates_dir = session_dir / "document_updates"

        if not updates_dir.exists():
            return {}

        updates: Dict[str, List[Dict]] = {}
        for file in sorted(updates_dir.glob("*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_name = data.get("document_name", file.stem)
            updates.setdefault(doc_name, []).append(data)

        return updates

    def save_clusters(self, session_name: str, clusters: list) -> Path:
        """Serialize DocumentCluster list to JSON (Path objects stored as strings)."""
        session_dir = self.create_session(session_name)
        data = []
        for cl in clusters:
            data.append({
                "cluster_id": cl.cluster_id,
                "cluster_name": cl.cluster_name,
                "cluster_type": cl.cluster_type,
                "entry_point": cl.entry_point,
                "cobol_program_names": sorted(cl.cobol_program_names),
                "cobol_files": [str(p) for p in cl.cobol_files],
                "doc_files": [str(p) for p in cl.doc_files],
                "shared_doc_files": [str(p) for p in cl.shared_doc_files],
            })
        file_path = session_dir / "clusters.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return file_path

    def load_clusters(self, session_name: str) -> list:
        """Restore DocumentCluster list; paths that no longer exist are silently dropped."""
        from src.pipeline.cluster_builder import DocumentCluster
        session_dir = self.storage_dir / session_name
        file_path = session_dir / "clusters.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            clusters = []
            for d in data:
                cobol  = [Path(p) for p in d.get("cobol_files", [])      if Path(p).exists()]
                docs   = [Path(p) for p in d.get("doc_files", [])        if Path(p).exists()]
                shared = [Path(p) for p in d.get("shared_doc_files", []) if Path(p).exists()]
                clusters.append(DocumentCluster(
                    cluster_id=d["cluster_id"],
                    cluster_name=d["cluster_name"],
                    cluster_type=d["cluster_type"],
                    entry_point=d.get("entry_point"),
                    cobol_program_names=set(d.get("cobol_program_names", [])),
                    cobol_files=cobol,
                    doc_files=docs,
                    shared_doc_files=shared,
                ))
            return clusters
        except Exception:
            return []

    def save_cluster_summaries(self, session_name: str, summaries: list) -> Path:
        """Serialize ClusterSummary list to JSON, preserving edge_cases and source_files."""
        session_dir = self.create_session(session_name)
        data = []
        for cs in summaries:
            data.append({
                "cluster_id": getattr(cs, "cluster_id", ""),
                "cluster_name": cs.cluster_name,
                "cluster_type": getattr(cs, "cluster_type", ""),
                "summary": cs.summary,
                "technical_summary": getattr(cs, "technical_summary", ""),
                "key_processes": cs.key_processes,
                "systems_mentioned": cs.systems_mentioned,
                "file_count": cs.file_count,
                "entry_point": getattr(cs, "entry_point", None),
                "edge_cases": getattr(cs, "edge_cases", []),
                "source_files": getattr(cs, "source_files", []),
            })
        file_path = session_dir / "cluster_summaries.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return file_path

    def load_cluster_summaries(self, session_name: str) -> list:
        """Restore ClusterSummary list."""
        from src.pipeline.hierarchical_summarizer import ClusterSummary
        session_dir = self.storage_dir / session_name
        file_path = session_dir / "cluster_summaries.json"
        if not file_path.exists():
            return []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            summaries = []
            for d in data:
                summaries.append(ClusterSummary(
                    cluster_id=d.get("cluster_id", ""),
                    cluster_name=d["cluster_name"],
                    cluster_type=d.get("cluster_type", ""),
                    summary=d["summary"],
                    technical_summary=d.get("technical_summary", ""),
                    key_processes=d.get("key_processes", []),
                    systems_mentioned=d.get("systems_mentioned", []),
                    file_count=d.get("file_count", 0),
                    entry_point=d.get("entry_point"),
                    edge_cases=d.get("edge_cases", []),
                    source_files=d.get("source_files", []),
                ))
            return summaries
        except Exception:
            return []

    def delete_session(self, session_name: str) -> None:
        """Delete a session and all its files."""
        session_dir = self.storage_dir / session_name

        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
