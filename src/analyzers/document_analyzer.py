"""Document analysis engines."""

from typing import List, Dict, Optional
from dataclasses import dataclass
from src.document_loaders import DocumentContent
from src.llm_integration import AzureLLMClient, PromptBuilder


@dataclass
class AnalysisResult:
    """Results from document analysis."""
    document_name: str
    summary: str
    key_processes: List[str]
    systems_mentioned: List[str]
    technical_details: List[str]


@dataclass
class ProcessDocument:
    """Generated process document."""
    overview: str
    integrated_processes: str
    dependencies: str
    data_flow: str
    decision_points: str
    systems_and_components: str
    appendix: str = ""
    process_flow_diagram: str = ""  # Mermaid flowchart syntax (rendered in-browser)
    process_flow_ascii: str = ""   # Plain-text flowchart for print / Word export


@dataclass
class GapAnalysis:
    """Gap analysis results."""
    missing_steps: List[str]
    undefined_dependencies: List[str]
    incomplete_transformations: List[str]
    missing_integrations: List[str]
    error_handling_gaps: List[str]
    security_gaps: List[str]
    resource_gaps: List[str]


class DocumentAnalyzer:
    """Analyzes individual documents using LLM."""

    def __init__(self):
        self.llm = AzureLLMClient()

    def analyze_document(self, doc: DocumentContent) -> AnalysisResult:
        """Analyze a single document and extract key information."""
        prompt = PromptBuilder.build_document_summary_prompt(doc.content)
        summary = self.llm.query(prompt)

        return AnalysisResult(
            document_name=doc.filename,
            summary=summary,
            key_processes=self._extract_list(summary, "process", 5),
            systems_mentioned=self._extract_list(summary, "system", 5),
            technical_details=self._extract_list(summary, "technical", 5)
        )

    @staticmethod
    def _extract_list(text: str, category: str, limit: int = 5) -> List[str]:
        """Extract bullet items under the labeled section matching `category`."""
        # Map category keywords to the section headers used in the prompt
        header_map = {
            "process": ["KEY PROCESSES"],
            "system": ["SYSTEMS MENTIONED"],
            "technical": ["TECHNICAL DETAILS"],
        }
        target_headers = header_map.get(category.lower(), [category.upper()])

        lines = text.split('\n')
        in_section = False
        items: List[str] = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper().rstrip(':')

            # Detect section header
            if stripped.endswith(':') or (stripped.isupper() and len(stripped) > 3):
                in_section = any(h in upper for h in target_headers)
                continue

            if in_section:
                # Stop at the next section header
                if stripped and stripped[0].isupper() and stripped.endswith(':'):
                    break
                item = stripped.lstrip('-•*+ ').strip()
                # Skip template artifact lines from prompt format
                if item.startswith('(bullet:') or item in ('...', '….') or item.startswith('[') or item.startswith('List each'):
                    continue
                if len(item) > 5:
                    items.append(item)
                    if len(items) >= limit:
                        break

        # Fallback: grab bullet lines anywhere in the text
        if not items:
            for line in lines:
                item = line.strip().lstrip('-•*+ ').strip()
                if len(item) > 10:
                    items.append(item)
                    if len(items) >= limit:
                        break

        return items[:limit] if items else [f"No {category} details extracted"]


class ProcessDocumentBuilder:
    """Builds comprehensive process documents from analysis results."""

    _SECTIONS = [
        "overview",
        "integrated_processes",
        "dependencies",
        "data_flow",
        "decision_points",
        "systems_and_components",
        "appendix",
        "process_flow_diagram",
    ]
    _TOKENS_PER_SECTION = 40000

    def __init__(self):
        self.llm = AzureLLMClient()

    def build_from_cluster_summaries(
        self,
        cluster_summaries: list,  # list[ClusterSummary] — imported lazily to avoid circular
        progress_callback=None,   # optional: called with (section_label, idx, total)
        context_block: str = "",  # from ProcessContext.to_prompt_block()
    ) -> "ProcessDocument":
        """Build process document from hierarchical cluster summaries (bulk mode).

        Uses one LLM call per section so reasoning-model token budgets are not
        exhausted before the final sections are written.
        """
        summaries = [cs.summary for cs in cluster_summaries]
        return self._build_section_by_section(
            lambda key: PromptBuilder.build_section_prompt_from_clusters(key, summaries, context_block),
            progress_callback,
        )

    def build_process_document(
        self,
        analyses: List[AnalysisResult],
        progress_callback=None,  # optional: called with (section_label, idx, total)
        context_block: str = "",  # from ProcessContext.to_prompt_block()
    ) -> "ProcessDocument":
        """Build comprehensive process document from multiple analyses.

        Uses one LLM call per section so reasoning-model token budgets are not
        exhausted before the final sections are written.
        """
        summaries = [analysis.summary for analysis in analyses]
        return self._build_section_by_section(
            lambda key: PromptBuilder.build_section_prompt_from_analyses(key, summaries, context_block),
            progress_callback,
        )

    def _build_section_by_section(self, prompt_fn, progress_callback=None) -> "ProcessDocument":
        """Call the LLM once per section and assemble the results."""
        results: Dict[str, str] = {}
        total = len(self._SECTIONS)
        for idx, section_key in enumerate(self._SECTIONS):
            if progress_callback:
                label = section_key.replace("_", " ").title()
                progress_callback(label, idx + 1, total)
            prompt = prompt_fn(section_key)
            results[section_key] = self.llm.query(prompt, max_tokens=self._TOKENS_PER_SECTION)
        return ProcessDocument(
            overview=results.get("overview", ""),
            integrated_processes=results.get("integrated_processes", ""),
            dependencies=results.get("dependencies", ""),
            data_flow=results.get("data_flow", ""),
            decision_points=results.get("decision_points", ""),
            systems_and_components=results.get("systems_and_components", ""),
            appendix=results.get("appendix", ""),
            process_flow_diagram=results.get("process_flow_diagram", ""),
            process_flow_ascii=results.get("process_flow_ascii", ""),
        )

class GapAnalyzer:
    """Identifies gaps and missing information in process documents."""

    def __init__(self):
        self.llm = AzureLLMClient()

    def analyze_gaps(
        self,
        process_document: ProcessDocument,
        context_block: str = "",
    ) -> GapAnalysis:
        """Analyze process document for gaps and missing information."""
        doc_text = f"""
Overview: {process_document.overview}
Processes: {process_document.integrated_processes}
Dependencies: {process_document.dependencies}
Data Flow: {process_document.data_flow}
Decision Points: {process_document.decision_points}
Systems: {process_document.systems_and_components}
"""

        prompt = PromptBuilder.build_gap_analysis_prompt(doc_text)
        if context_block:
            prompt = context_block + "\n\n" + prompt
        gap_response = self.llm.query(prompt)

        return self._parse_gaps(gap_response)

    @staticmethod
    def _parse_gaps(text: str) -> GapAnalysis:
        """Parse gap analysis response into structured data."""
        return GapAnalysis(
            missing_steps=GapAnalyzer._extract_category(text, "missing", 5),
            undefined_dependencies=GapAnalyzer._extract_category(text, "undefined", 5),
            incomplete_transformations=GapAnalyzer._extract_category(text, "transformation", 5),
            missing_integrations=GapAnalyzer._extract_category(text, "integration", 5),
            error_handling_gaps=GapAnalyzer._extract_category(text, "error", 5),
            security_gaps=GapAnalyzer._extract_category(text, "security", 5),
            resource_gaps=GapAnalyzer._extract_category(text, "resource", 5)
        )

    @staticmethod
    def _extract_category(text: str, category: str, limit: int) -> List[str]:
        """Extract bullet items under the gap section matching `category`."""
        # Map each category to the section header keywords used in the prompt
        header_map = {
            "missing": ["MISSING STEPS"],
            "undefined": ["UNDEFINED DEPENDENCIES"],
            "transformation": ["INCOMPLETE TRANSFORMATIONS"],
            "integration": ["MISSING INTEGRATIONS"],
            "error": ["ERROR HANDLING GAPS"],
            "security": ["SECURITY GAPS"],
            "resource": ["RESOURCE GAPS"],
        }
        target_headers = header_map.get(category.lower(), [category.upper()])

        lines = text.split('\n')
        in_section = False
        items: List[str] = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper().rstrip(':')

            # Detect section header (ends with colon or is ALL CAPS)
            if stripped.endswith(':') and len(stripped) > 3:
                in_section = any(h in upper for h in target_headers)
                continue

            if in_section:
                item = stripped.lstrip('-•*+ ').strip()
                # Skip template artifact lines from prompt format
                if item.startswith('(bullet:') or item in ('...', '….') or item.startswith('[') or item.startswith('List each'):
                    continue
                if len(item) > 5:
                    items.append(item)
                    if len(items) >= limit:
                        break
                elif not stripped:
                    pass  # skip blank lines within section
                elif stripped[0].isupper() and stripped.endswith(':'):
                    break  # hit next section

        return items[:limit] if items else [f"No {category} gaps identified"]


class ClarificationQuestionGenerator:
    """Generates clarification questions to enhance documentation."""

    def __init__(self):
        self.llm = AzureLLMClient()

    def generate_questions(
        self,
        process_document: ProcessDocument,
        gap_analysis: GapAnalysis
    ) -> List[Dict[str, str]]:
        """Generate clarification questions based on document and gaps."""
        doc_text = self._format_process_document(process_document)
        gap_text = self._format_gap_analysis(gap_analysis)

        prompt = PromptBuilder.build_clarification_questions_prompt(doc_text, gap_text)
        response = self.llm.query(prompt)

        return self._parse_questions(response)

    @staticmethod
    def _format_process_document(doc: ProcessDocument) -> str:
        """Format process document for LLM."""
        return f"""
Overview: {doc.overview}
Integrated Processes: {doc.integrated_processes}
Dependencies: {doc.dependencies}
Data Flow: {doc.data_flow}
Decision Points: {doc.decision_points}
Systems: {doc.systems_and_components}
"""

    @staticmethod
    def _format_gap_analysis(gaps: GapAnalysis) -> str:
        """Format gap analysis for LLM."""
        return f"""
Missing Steps: {', '.join(gaps.missing_steps)}
Undefined Dependencies: {', '.join(gaps.undefined_dependencies)}
Incomplete Transformations: {', '.join(gaps.incomplete_transformations)}
Missing Integrations: {', '.join(gaps.missing_integrations)}
Error Handling Gaps: {', '.join(gaps.error_handling_gaps)}
Security Gaps: {', '.join(gaps.security_gaps)}
Resource Gaps: {', '.join(gaps.resource_gaps)}
"""

    @staticmethod
    def _parse_questions(text: str) -> List[Dict[str, str]]:
        """Parse generated questions into structured format."""
        import re
        questions = []
        current_q: Optional[Dict[str, str]] = None

        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Match any numbered prefix (1. 2. 10. 15.) or bullet markers
            numbered = re.match(r'^(\d{1,2})[.)]\s+(.+)', line)
            bulleted = re.match(r'^[-*•]\s+(.+)', line)

            if numbered or bulleted:
                if current_q:
                    questions.append(current_q)
                question_text = numbered.group(2) if numbered else bulleted.group(1)
                current_q = {'question': question_text.strip(), 'rationale': ''}
            elif current_q:
                low = line.lower()
                if any(kw in low for kw in ('why', 'importance', 'rationale', 'reason', 'because')):
                    current_q['rationale'] = line.lstrip('-•*: ').strip()
                elif not current_q['question'].endswith('?') and len(line) > 10:
                    # Continuation of a multi-line question
                    current_q['question'] += ' ' + line

        if current_q:
            questions.append(current_q)

        return questions[:15]
