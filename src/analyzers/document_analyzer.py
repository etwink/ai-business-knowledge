"""Document analysis engines."""

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from src.document_loaders import DocumentContent
from src.llm_integration import AzureLLMClient, PromptBuilder

# Broad pattern catches well-formed [ref:3] AND malformed [ref:3: extra text]
_REF_RE = re.compile(r'\[ref:(\d+)[^\]]*\]', re.IGNORECASE)

_CHUNK_CHARS = 5_000
_MAX_FILES_PER_CLUSTER = 15
_VERIFY_WORKERS = 3

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "in", "of", "to", "and", "or", "for", "with",
    "that", "this", "it", "be", "are", "was", "not", "has", "have", "does",
    "do", "on", "at", "by", "from", "as", "its", "which", "what", "how",
    "when", "where", "who", "will", "can", "may", "should", "no", "any",
    "all", "there", "their", "they", "we", "our", "if", "but", "so",
})

# Maps ProcessDocument field names → keywords that suggest a gap belongs there
_DOC_SECTION_KEYWORDS: dict[str, list[str]] = {
    "data_flow": ["data", "flow", "input", "output", "transfer", "feed", "pipe", "file", "record"],
    "decision_points": ["decision", "rule", "condition", "branch", "logic", "validate", "check"],
    "dependencies": ["depend", "require", "external", "interface", "api", "call", "invoke", "library"],
    "integrated_processes": ["process", "workflow", "step", "procedure", "integration", "sequence"],
    "systems_and_components": ["system", "component", "module", "service", "application", "program"],
    "overview": ["overview", "purpose", "objective", "goal", "scope", "background"],
}


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


    def _find_gap_cluster(
        self,
        gap_desc: str,
        process_doc: ProcessDocument,
        cluster_summaries: list,
    ):
        """Return the ClusterSummary most relevant to gap_desc, or None."""
        if not cluster_summaries:
            return None

        # Citation markers take priority: [ref:N] or bare [N] in the gap text
        ref_nums = set(re.findall(r'\[(?:ref:)?(\d+)\]', gap_desc, re.IGNORECASE))
        if ref_nums and process_doc.citations:
            for rn in ref_nums:
                key = f"ref:{rn}"
                if key in process_doc.citations:
                    target_name = process_doc.citations[key].get("cluster_name", "")
                    for cs in cluster_summaries:
                        if cs.cluster_name == target_name:
                            return cs

        # Keyword overlap fallback
        tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', gap_desc.lower())) - _STOP_WORDS
        if not tokens:
            return cluster_summaries[0]

        best, best_score = None, -1
        for cs in cluster_summaries:
            cs_text = (
                cs.cluster_name + " "
                + cs.summary
                + " " + " ".join(cs.key_processes)
            ).lower()
            score = sum(1 for t in tokens if t in cs_text)
            if score > best_score:
                best_score, best = score, cs
        return best

    def _load_cluster_source_content(
        self,
        cluster,
        file_path_map: dict,
    ) -> str:
        """Read up to _MAX_FILES_PER_CLUSTER source files, capped at _CHUNK_CHARS each."""
        parts: list[str] = []
        for fname in cluster.source_files[:_MAX_FILES_PER_CLUSTER]:
            path = file_path_map.get(fname)
            if path is None:
                continue
            try:
                text = Path(path).read_text(errors="replace")[:_CHUNK_CHARS]
                parts.append(f"--- {fname} ---\n{text}")
            except Exception:
                continue
        return "\n\n".join(parts)

    def _verify_cluster_gaps(
        self,
        cluster_name: str,
        source_content: str,
        gap_descs: list,
    ) -> list:
        """LLM call: for each gap, decide if it's a doc miss or a true gap."""
        gaps_block = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(gap_descs))
        prompt = (
            "You are reviewing source documents to determine whether documented gaps are "
            "true gaps or documentation misses.\n\n"
            f"Source files from cluster '{cluster_name}':\n"
            f"<source>\n{source_content}\n</source>\n\n"
            f"Gaps to verify:\n{gaps_block}\n\n"
            "For each gap decide:\n"
            "- is_doc_miss: true  → source files contain enough information to answer this gap "
            "(it was missed during summarization, not a true gap)\n"
            "- is_doc_miss: false → source files do NOT contain the information "
            "(genuine gap that a subject-matter expert must fill)\n"
            "- note: one sentence explaining your finding\n"
            "- patch_hint: if is_doc_miss=true, quote the relevant content from the source files "
            "(max 300 chars); otherwise empty string\n\n"
            "Return a JSON array with one object per gap in the same order:\n"
            '[{"index":1,"is_doc_miss":true,"note":"...","patch_hint":"..."},...]\n'
            "Return ONLY valid JSON."
        )
        _empty = [
            {"index": i + 1, "is_doc_miss": False, "note": "", "patch_hint": ""}
            for i in range(len(gap_descs))
        ]
        try:
            result = self.llm.query(
                prompt,
                max_tokens=min(6000, len(gap_descs) * 200 + 1000),
            )
            m = re.search(r'\[[\s\S]*\]', result)
            if not m:
                return _empty
            verdicts = json.loads(m.group())
            return verdicts if isinstance(verdicts, list) else _empty
        except Exception:
            return _empty

    def _patch_cluster_summary(self, cluster, missed_hints: list) -> None:
        """Update cluster.summary (and technical_summary for code clusters) in-place."""
        if not missed_hints:
            return
        hints_block = "\n".join(f"- {h}" for h in missed_hints if h)
        prompt = (
            "Update the cluster summary below to incorporate the following missed topics. "
            "Keep the same structure and tone. Add only what is missing.\n\n"
            f"Current summary:\n{cluster.summary}\n\n"
            f"Missed topics to incorporate:\n{hints_block}\n\n"
            "Return ONLY the updated summary text."
        )
        try:
            updated = self.llm.query(prompt, max_tokens=2000).strip()
            if updated:
                cluster.summary = updated
            if cluster.cluster_type in ("cobol", "mixed") and cluster.technical_summary:
                tech_prompt = (
                    "Update the technical summary below to incorporate the following missed topics.\n\n"
                    f"Current technical summary:\n{cluster.technical_summary}\n\n"
                    f"Missed topics:\n{hints_block}\n\n"
                    "Return ONLY the updated technical summary text."
                )
                updated_tech = self.llm.query(tech_prompt, max_tokens=1500).strip()
                if updated_tech:
                    cluster.technical_summary = updated_tech
        except Exception:
            pass

    def _patch_process_doc_section(
        self,
        process_doc: ProcessDocument,
        gap_desc: str,
        patch_hint: str,
    ) -> str:
        """Append patch_hint to the most relevant ProcessDocument section. Returns field name."""
        if not patch_hint:
            return ""
        tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', gap_desc.lower())) - _STOP_WORDS
        best_section = "appendix"
        best_score = 0
        for section, keywords in _DOC_SECTION_KEYWORDS.items():
            score = sum(1 for k in keywords if k in tokens)
            if score > best_score:
                best_score, best_section = score, section
        current = getattr(process_doc, best_section, "") or ""
        sep = "\n\n" if current else ""
        setattr(process_doc, best_section, current + sep + f"[Gap Verification] {patch_hint}")
        return best_section

    def verify_and_fix_gaps(
        self,
        gap_analysis: GapAnalysis,
        process_doc: ProcessDocument,
        cluster_summaries: list,
        file_path_map: dict,
    ) -> tuple:
        """Verify ranked gaps against actual source files; patch summaries and process doc.

        For each gap, locates the most relevant cluster, loads its source files,
        and asks the LLM whether the gap is a true gap or a documentation miss.
        Misses are patched into the cluster summary and process document in-place.

        Returns:
            (GapAnalysis, ProcessDocument, list[ClusterSummary], int)
            where int is the number of doc-miss fixes applied.
        Falls back gracefully on any error — never raises.
        """
        if not gap_analysis.ranked_gaps or not cluster_summaries:
            return gap_analysis, process_doc, cluster_summaries, 0

        # Map each gap to its most relevant cluster
        cluster_gap_groups: dict[str, list] = defaultdict(list)
        cluster_by_id: dict[str, object] = {}
        for i, gap in enumerate(gap_analysis.ranked_gaps):
            cs = self._find_gap_cluster(gap["description"], process_doc, cluster_summaries)
            if cs is not None:
                cluster_gap_groups[cs.cluster_id].append((i, gap["description"]))
                cluster_by_id[cs.cluster_id] = cs

        updated_gaps = list(gap_analysis.ranked_gaps)
        fix_count = 0

        def _process_cluster(cluster_id: str) -> dict:
            cs = cluster_by_id[cluster_id]
            source_content = self._load_cluster_source_content(cs, file_path_map)
            if not source_content:
                return {"cluster_id": cluster_id, "verdicts": []}
            gap_descs = [d for _, d in cluster_gap_groups[cluster_id]]
            verdicts = self._verify_cluster_gaps(cs.cluster_name, source_content, gap_descs)
            return {
                "cluster_id": cluster_id,
                "cluster": cs,
                "gaps": cluster_gap_groups[cluster_id],
                "verdicts": verdicts,
            }

        with ThreadPoolExecutor(max_workers=_VERIFY_WORKERS) as executor:
            futures = {
                executor.submit(_process_cluster, cid): cid
                for cid in cluster_gap_groups
            }
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass

        for result in results:
            verdicts = result.get("verdicts") or []
            if not verdicts:
                continue
            cs = result.get("cluster")
            gaps = result.get("gaps", [])
            missed_hints: list[str] = []

            for (gap_idx, gap_desc), verdict in zip(gaps, verdicts):
                is_miss = bool(verdict.get("is_doc_miss", False))
                note = str(verdict.get("note", "")).strip()
                hint = str(verdict.get("patch_hint", "")).strip()
                updated_gaps[gap_idx] = {
                    **updated_gaps[gap_idx],
                    "is_doc_miss": is_miss,
                    "verification_note": note,
                }
                if is_miss and hint:
                    missed_hints.append(hint)
                    self._patch_process_doc_section(process_doc, gap_desc, hint)
                    fix_count += 1

            if cs is not None and missed_hints:
                self._patch_cluster_summary(cs, missed_hints)

        updated_analysis = GapAnalysis(
            gaps_by_category=gap_analysis.gaps_by_category,
            edge_cases=gap_analysis.edge_cases,
            resource_gaps=gap_analysis.resource_gaps,
            ranked_gaps=updated_gaps,
        )
        return updated_analysis, process_doc, cluster_summaries, fix_count


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
