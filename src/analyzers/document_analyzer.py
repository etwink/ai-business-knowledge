"""Document analysis engines."""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from src.document_loaders import DocumentContent
from src.llm_integration import AzureLLMClient, PromptBuilder

# Broad pattern catches well-formed [ref:3] AND malformed [ref:3: extra text]
_REF_RE = re.compile(r'\[ref:(\d+)[^\]]*\]', re.IGNORECASE)


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
    # Maps "ref:N" → {"cluster_name": str, "files": list[str]}.
    # Populated only when built from cluster summaries; empty for the analyses path.
    citations: dict = field(default_factory=dict)


@dataclass
class GapAnalysis:
    """
    Dynamic, audience-aware gap analysis results.

    gaps_by_category : audience-specific category names → list of gap strings.
                       Categories vary per audience (e.g. developer vs. business).
    edge_cases       : always present — boundary conditions, unusual inputs,
                       failure scenarios, and unaddressed exception paths.
    resource_gaps    : always present — missing role assignments, undefined
                       ownership, or unidentified responsible teams.
    ranked_gaps      : all gaps sorted by importance (1–10) after the ranking pass.
                       Each entry is a dict with keys: description, category, importance.
    """
    gaps_by_category: Dict[str, List[str]] = field(default_factory=dict)
    edge_cases: List[str] = field(default_factory=list)
    resource_gaps: List[str] = field(default_factory=list)
    ranked_gaps: List[Dict] = field(default_factory=list)


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

    _MERMAID_REPAIR_PROMPT = (
        "The Mermaid flowchart below has a syntax error and will not render. "
        "Rewrite it as valid Mermaid flowchart TD syntax that describes the same process.\n\n"
        "Strict rules:\n"
        "- First line MUST be exactly: flowchart TD\n"
        "- Process steps: A[Step Name]\n"
        "- Decision points: B{Condition}\n"
        "- Data stores: C[(Name)]\n"
        "- External systems: D([Name])\n"
        "- Arrows: --> and -->|label|\n"
        "- Node IDs: short alphanumeric tokens only (A, B, PROC1) — NO hyphens, NO spaces\n"
        "- Labels must NOT contain: parentheses (), quotes, angle brackets <>, or unmatched braces\n"
        "- Keep it to 10–20 nodes for readability\n"
        "- Output ONLY the corrected Mermaid syntax — no markdown fences, no explanation, no prose.\n\n"
        "Broken code:\n"
    )

    @staticmethod
    def _validate_mermaid(code: str) -> bool:
        """Return True if code looks like structurally valid Mermaid flowchart syntax."""
        if not code or not code.strip():
            return False
        stripped = code.strip()
        for fence in ("```mermaid", "```"):
            if stripped.startswith(fence):
                stripped = stripped[len(fence):]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
        first_line = stripped.split('\n')[0].strip()
        if not re.match(r'^(flowchart|graph)\s+(TD|LR|TB|BT|RL)\b', first_line, re.IGNORECASE):
            return False
        content_lines = [l for l in stripped.split('\n') if l.strip() and not l.strip().startswith('%%')]
        return len(content_lines) >= 3

    def _repair_mermaid(self, bad_code: str) -> str:
        """Ask the LLM to fix broken Mermaid syntax. Returns the repaired code."""
        return self.llm.query(self._MERMAID_REPAIR_PROMPT + bad_code, max_tokens=4000)

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
        exhausted before the final sections are written.  After all sections are
        generated, [ref:N] citation markers are extracted into ProcessDocument.citations
        and stripped from the body text so the final output is clean.
        """
        summaries = [cs.summary for cs in cluster_summaries]
        tech_summaries = [
            getattr(cs, "technical_summary", "") for cs in cluster_summaries
        ]
        doc = self._build_section_by_section(
            lambda key: PromptBuilder.build_section_prompt_from_clusters(
                key, summaries, context_block, tech_summaries=tech_summaries
            ),
            progress_callback,
        )

        # Build 1-based cluster source map from ClusterSummary.source_files
        cluster_sources: dict[int, dict] = {
            i + 1: {
                "cluster_name": cs.cluster_name,
                "files": getattr(cs, "source_files", []),
            }
            for i, cs in enumerate(cluster_summaries)
        }
        return self._extract_citations_from_document(doc, cluster_sources)

    def _build_section_by_section(self, prompt_fn, progress_callback=None) -> "ProcessDocument":
        """Call the LLM once per section and assemble the results."""
        results: Dict[str, str] = {}
        total = len(self._SECTIONS)
        for idx, section_key in enumerate(self._SECTIONS):
            if progress_callback:
                label = section_key.replace("_", " ").title()
                progress_callback(label, idx + 1, total)
            prompt = prompt_fn(section_key)
            output = self.llm.query(prompt, max_tokens=self._TOKENS_PER_SECTION)
            if section_key == "process_flow_diagram" and not self._validate_mermaid(output):
                if progress_callback:
                    progress_callback("Process Flow Diagram (repairing syntax)", idx + 1, total)
                output = self._repair_mermaid(output)
            results[section_key] = output
        return ProcessDocument(
            **{k: results.get(k, "") for k in self._SECTIONS},
            process_flow_ascii="",
            citations={},
        )

    @staticmethod
    def _extract_citations_from_document(
        doc: "ProcessDocument",
        cluster_sources: Dict[int, dict],
    ) -> "ProcessDocument":
        """Parse [ref:N] markers from narrative sections, build citations dict, convert to [N].

        cluster_sources: {N: {"cluster_name": str, "files": list[str]}}
        Handles malformed markers like [ref:3: extra text] by matching [ref:N<anything>].
        Returns a new ProcessDocument with [ref:N] → [N] inline and citations populated.
        """
        _NARRATIVE = (
            "overview", "integrated_processes", "dependencies",
            "data_flow", "decision_points", "systems_and_components",
        )

        # Collect every cluster index referenced across all narrative sections
        referenced: set[int] = set()
        for field_name in _NARRATIVE:
            for m in _REF_RE.finditer(getattr(doc, field_name, "")):
                referenced.add(int(m.group(1)))

        if not referenced:
            return doc  # LLM emitted no markers — nothing to convert

        # Build citations dict
        citations: dict = {
            f"ref:{n}": cluster_sources[n]
            for n in sorted(referenced)
            if n in cluster_sources
        }

        # Convert [ref:N...] → [N] so inline anchors remain visible in the document
        def _clean(text: str) -> str:
            return _REF_RE.sub(lambda m: f"[{m.group(1)}]", text).strip()

        return ProcessDocument(
            overview=_clean(doc.overview),
            integrated_processes=_clean(doc.integrated_processes),
            dependencies=_clean(doc.dependencies),
            data_flow=_clean(doc.data_flow),
            decision_points=_clean(doc.decision_points),
            systems_and_components=_clean(doc.systems_and_components),
            appendix=doc.appendix,
            process_flow_diagram=doc.process_flow_diagram,
            process_flow_ascii=doc.process_flow_ascii,
            citations=citations,
        )

_RANKING_SYSTEM = (
    "You are a business risk analyst. Rank the provided gaps by how critical they are "
    "to address. Return ONLY a JSON array of integers."
)

# Edge cases and error-handling gaps are rare by definition; cap their scores
# so they don't crowd out structural or compliance gaps in the ranked list.
_EDGE_ERROR_CAP = 6
_EDGE_ERROR_KEYWORDS = ("edge", "error", "exception")


class GapAnalyzer:
    """
    Identifies gaps in a process document using audience-specific prompts,
    then ranks every gap by business importance in a second LLM pass.
    """

    def __init__(self):
        self.llm = AzureLLMClient()

    def analyze_gaps(
        self,
        process_document: ProcessDocument,
        context_block: str = "",
        audience_key: str = "new_employee",
        audience_note: str = "",
        cluster_edge_cases: Optional[List[str]] = None,
    ) -> GapAnalysis:
        """
        Two-pass gap analysis:
          Pass 1 — identify all gaps in audience-specific categories plus
                   edge cases and resource gaps (JSON output).
          Pass 2 — rank every identified gap 1–10 by business importance.
        """
        doc_text = (
            f"Overview: {process_document.overview}\n"
            f"Processes: {process_document.integrated_processes}\n"
            f"Dependencies: {process_document.dependencies}\n"
            f"Data Flow: {process_document.data_flow}\n"
            f"Decision Points: {process_document.decision_points}\n"
            f"Systems: {process_document.systems_and_components}\n"
        )

        prompt = PromptBuilder.build_gap_analysis_prompt(
            doc_text,
            audience_key=audience_key,
            audience_note=audience_note,
            cluster_edge_cases=cluster_edge_cases,
        )
        if context_block:
            prompt = context_block + "\n\n" + prompt

        gap_response = self.llm.query(prompt, max_tokens=min(8000, max(4000, len(doc_text) // 8)))
        gaps_by_category, edge_cases, resource_gaps = self._parse_gap_response(gap_response)

        # Build flat list of all gaps for the ranking pass
        all_gaps: List[Dict] = [
            {"category": cat, "description": desc}
            for cat, descs in gaps_by_category.items()
            for desc in descs
        ] + [
            {"category": "Edge Cases", "description": ec}
            for ec in edge_cases
        ] + [
            {"category": "Resource Gaps", "description": rg}
            for rg in resource_gaps
        ]

        ranked_gaps = self._rank_gaps(all_gaps)

        return GapAnalysis(
            gaps_by_category=gaps_by_category,
            edge_cases=edge_cases,
            resource_gaps=resource_gaps,
            ranked_gaps=ranked_gaps,
        )

    # ------------------------------------------------------------------
    # Pass 1 — parse JSON gap response
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gap_response(text: str) -> tuple:
        """
        Parse the JSON gap-analysis response.
        Returns (gaps_by_category, edge_cases, resource_gaps).
        Falls back to empty structures if JSON cannot be parsed.
        """
        m = re.search(r'\{[\s\S]*\}', text)
        if not m:
            return {}, [], []

        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            # Try stripping trailing commas before the closing bracket/brace
            try:
                cleaned = re.sub(r',(\s*[}\]])', r'\1', m.group())
                data = json.loads(cleaned)
            except Exception:
                return {}, [], []

        gaps_by_category: Dict[str, List[str]] = {}
        for gap in data.get("gaps", []):
            cat = str(gap.get("category", "Other")).strip()
            desc = str(gap.get("description", "")).strip()
            if cat and desc:
                gaps_by_category.setdefault(cat, []).append(desc)

        edge_cases = [str(e).strip() for e in data.get("edge_cases", []) if str(e).strip()]
        resource_gaps = [str(r).strip() for r in data.get("resource_gaps", []) if str(r).strip()]

        return gaps_by_category, edge_cases, resource_gaps

    # ------------------------------------------------------------------
    # Pass 2 — importance ranking
    # ------------------------------------------------------------------

    def _rank_gaps(self, gaps: List[Dict]) -> List[Dict]:
        """
        Ask the LLM to rate each gap 1–10 by business criticality.
        Returns all gaps sorted descending by importance.
        Falls back to importance=5 for everything on any error.
        """
        if not gaps:
            return []

        lines = "\n".join(
            f"Gap Number: [{i+1}]\tGap Category: [{g['category']}]\tGap Description: {g['description']}"
            for i, g in enumerate(gaps)
        )
        prompt = (
            "Rate each gap 1–10 for business criticality:\n"
            "10 = Critical (blocks operations, security breach, data loss)\n"
            "7–9 = High (significant process failure risk, compliance issue)\n"
            "4–6 = Medium (process inefficiency, documentation clarity issue)\n"
            "1–3 = Low (minor improvement, cosmetic)\n\n"
            "IMPORTANT: Edge case and error/exception handling gaps should score no higher than "
            f"{_EDGE_ERROR_CAP} unless the scenario would cause a major cascading system failure. "
            "These scenarios are rare by definition and should not crowd out structural or compliance gaps.\n\n"
            f"Gaps:\n{lines}\n\n"
            "Return ONLY a JSON integer array ordered by Gap Number (where the rating of Gap Number 1 is at index = 0 in the JSON integer array, rating of Gap Number 2 is at index = 1, etc.). "
            "Example for 4 gaps: [8, 3, 7, 5]"
        )

        try:
            result = self.llm.query(prompt, system_message=_RANKING_SYSTEM, max_tokens=(len(gaps)*200)+1000)  # heuristic: allow ~200 tokens per gap plus some buffer for the prompt and instructions
            # Walk every [...] match in the response — the non-greedy regex may grab
            # short spurious brackets (e.g. "[15] gaps") before the actual array.
            for raw_match in re.findall(r'\[[^\[\]]*\]', result):
                try:
                    scores = json.loads(raw_match)
                except Exception:
                    continue
                if (
                    isinstance(scores, list)
                    and len(scores) == len(gaps)
                    and all(isinstance(s, (int, float)) for s in scores)
                ):
                    ranked = [
                        {**g, "importance": min(
                            max(1, min(10, int(float(s)))),
                            _EDGE_ERROR_CAP if any(
                                k in g["category"].lower()
                                for k in _EDGE_ERROR_KEYWORDS
                            ) else 10,
                        )}
                        for g, s in zip(gaps, scores)
                    ]
                    return sorted(ranked, key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass

        # Fallback: neutral importance
        return sorted(
            [{**g, "importance": 5} for g in gaps],
            key=lambda x: x["importance"],
            reverse=True,
        )


    def verify_gaps_against_summaries(
        self,
        gap_analysis: GapAnalysis,
        cluster_summaries: list,
    ) -> GapAnalysis:
        """Cross-check ranked gaps against original cluster summaries.

        A gap that is addressed in the source summaries is a documentation miss —
        the source material covers it but the process document did not capture it.
        A gap absent from the source summaries is a true gap the SME must fill.

        Tags each entry in ranked_gaps with:
          ``is_doc_miss`` (bool) — True if the source summaries cover this gap.
          ``verification_note`` (str) — one-sentence explanation.

        Returns a new GapAnalysis with those fields set on ranked_gaps.
        Falls back gracefully (no changes) on any LLM or parsing error.
        """
        if not gap_analysis.ranked_gaps or not cluster_summaries:
            return gap_analysis

        # Build a concise source-material block (cap each summary to avoid huge prompts)
        summaries_block = "\n\n".join(
            f"=== {getattr(cs, 'cluster_name', str(i))} ===\n"
            f"{getattr(cs, 'summary', '')[:1500]}"
            for i, cs in enumerate(cluster_summaries[:10])
        )

        gaps_block = "\n".join(
            f"[{i + 1}] {g['description']}"
            for i, g in enumerate(gap_analysis.ranked_gaps)
        )

        prompt = (
            "Below are summaries of the original source documents, followed by documentation "
            "gaps found in a generated process document.\n\n"
            "For each gap decide: does the source material above actually address or cover "
            "this topic with substantive detail?\n"
            "  - covered_in_source = true  → the source docs have the information; the process "
            "document generation simply missed capturing it (documentation miss).\n"
            "  - covered_in_source = false → the information is genuinely absent from the source "
            "docs and a subject-matter expert must fill it in.\n\n"
            f"Source Document Summaries:\n{summaries_block}\n\n"
            f"Gaps:\n{gaps_block}\n\n"
            "Return a JSON array with one object per gap, in the same order:\n"
            '[{"index":1,"covered_in_source":true,"note":"one sentence reason"},...]\n'
            "Return ONLY valid JSON."
        )

        try:
            result = self.llm.query(
                prompt,
                max_tokens=min(4000, len(gap_analysis.ranked_gaps) * 120 + 500),
            )
            m = re.search(r'\[[\s\S]*\]', result)
            if not m:
                return gap_analysis
            verdicts = json.loads(m.group())
            if not isinstance(verdicts, list) or len(verdicts) != len(gap_analysis.ranked_gaps):
                return gap_analysis

            updated = []
            for g, v in zip(gap_analysis.ranked_gaps, verdicts):
                updated.append({
                    **g,
                    "is_doc_miss": bool(v.get("covered_in_source", False)),
                    "verification_note": str(v.get("note", "")).strip(),
                })
            return GapAnalysis(
                gaps_by_category=gap_analysis.gaps_by_category,
                edge_cases=gap_analysis.edge_cases,
                resource_gaps=gap_analysis.resource_gaps,
                ranked_gaps=updated,
            )
        except Exception:
            return gap_analysis


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
        lines = []
        for cat, items in gaps.gaps_by_category.items():
            lines.append(f"{cat}: {', '.join(items)}")
        if gaps.edge_cases:
            lines.append(f"Edge Cases: {', '.join(gaps.edge_cases)}")
        if gaps.resource_gaps:
            lines.append(f"Resource Gaps: {', '.join(gaps.resource_gaps)}")
        return "\n".join(lines) if lines else "No gaps identified."

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
