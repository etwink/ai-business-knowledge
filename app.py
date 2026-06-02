"""Main Streamlit application for Document Analysis System."""

import streamlit as st
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Inject truststore into SSL context for better certificate handling (especially in enterprise environments)
import truststore
truststore.inject_into_ssl()

from src.document_loaders import get_loader
from src.analyzers import (
    DocumentAnalyzer,
    ProcessDocumentBuilder,
    GapAnalyzer,
    AnalysisResult,
    ProcessDocument,
    GapAnalysis
)
from src.utils import validate_file
from src.storage import AnalysisStorage
from src.pipeline import ConversationalAgent, DocumentUpdate, ProcessContextAgent
from src.pipeline.context_agent import AUDIENCE_LABELS, AUDIENCE_NOTES
from src.rag import KnowledgeBase, RAGAgent
import config


def initialize_session():
    """Initialize session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'analyses' not in st.session_state:
        st.session_state.analyses = []
    if 'process_document' not in st.session_state:
        st.session_state.process_document = None
    if 'gap_analysis' not in st.session_state:
        st.session_state.gap_analysis = None
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'storage' not in st.session_state:
        st.session_state.storage = AnalysisStorage()
    # Bulk mode state
    if 'bulk_scanned' not in st.session_state:
        st.session_state.bulk_scanned = None        # ScannedDocuments
    if 'bulk_clusters' not in st.session_state:
        st.session_state.bulk_clusters = None       # list[DocumentCluster]
    if 'bulk_cluster_summaries' not in st.session_state:
        st.session_state.bulk_cluster_summaries = None  # list[ClusterSummary]
    # Process context state (set before analysis to guide the LLM)
    if 'process_context' not in st.session_state:
        st.session_state.process_context = None    # ProcessContext | None
    # Chat agent state
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'document_updates' not in st.session_state:
        st.session_state.document_updates = {}  # doc_name -> list[DocumentUpdate]
    # Knowledge base / RAG chat state
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = None
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = []
    # Audience setting — controls writing style across all LLM outputs
    if 'doc_audience' not in st.session_state:
        st.session_state.doc_audience = "new_employee"


def _ensure_session() -> str:
    """Return current session name, auto-creating a timestamped one if not set."""
    if not st.session_state.current_session:
        st.session_state.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return st.session_state.current_session


def _get_context_block() -> str:
    """Return the full context block (audience note + process context) for LLM injection."""
    audience = st.session_state.get("doc_audience", "new_employee")
    audience_note = AUDIENCE_NOTES.get(audience, "")
    ctx_block = (
        st.session_state.process_context.to_prompt_block()
        if st.session_state.process_context and st.session_state.process_context.is_set()
        else ""
    )
    if audience_note and ctx_block:
        return audience_note + "\n\n" + ctx_block
    return audience_note or ctx_block


def render_sidebar():
    """Render sidebar navigation and settings."""
    with st.sidebar:
        st.title("📚 Document Analysis")

        st.subheader("Session Management")
        
        # Session creation/loading
        col1, col2 = st.columns([2, 1])
        with col1:
            session_name = st.text_input(
                "Session name",
                value=st.session_state.current_session or "analysis_session",
                placeholder="e.g., payroll_system"
            )
        with col2:
            if st.button("📁 New", help="Create or load session"):
                st.session_state.current_session = session_name
                st.rerun()
        
        if st.session_state.current_session:
            st.success(f"Session: **{st.session_state.current_session}**")
        else:
            st.caption("A session will be created automatically on the first LLM action.")
        
        # Load previous sessions
        existing_sessions = st.session_state.storage.list_sessions()
        if existing_sessions:
            st.subheader("Previous Sessions")
            selected_session = st.selectbox(
                "Load session",
                existing_sessions,
                key="session_selector"
            )
            if st.button("📂 Load Session"):
                try:
                    session_data = st.session_state.storage.load_session(selected_session)
                    st.session_state.current_session = selected_session
                    
                    # Populate session data
                    if 'analyses' in session_data:
                        # Reconstruct AnalysisResult objects
                        st.session_state.analyses = [
                            AnalysisResult(**analysis) for analysis in session_data['analyses']
                        ]
                    
                    if 'process_document' in session_data:
                        doc = session_data['process_document']
                        st.session_state.process_document = ProcessDocument(
                            overview=doc.get('overview', ''),
                            integrated_processes=doc.get('integrated_processes', ''),
                            dependencies=doc.get('dependencies', ''),
                            data_flow=doc.get('data_flow', ''),
                            decision_points=doc.get('decision_points', ''),
                            systems_and_components=doc.get('systems_and_components', ''),
                            appendix=doc.get('appendix', ''),
                            process_flow_diagram=doc.get('process_flow_diagram', ''),
                            process_flow_ascii=doc.get('process_flow_ascii', ''),
                        )
                    
                    if 'gap_analysis' in session_data:
                        gaps = session_data['gap_analysis']
                        st.session_state.gap_analysis = GapAnalysis(
                            missing_steps=gaps.get('missing_steps', []),
                            undefined_dependencies=gaps.get('undefined_dependencies', []),
                            incomplete_transformations=gaps.get('incomplete_transformations', []),
                            missing_integrations=gaps.get('missing_integrations', []),
                            error_handling_gaps=gaps.get('error_handling_gaps', []),
                            security_gaps=gaps.get('security_gaps', []),
                            resource_gaps=gaps.get('resource_gaps', [])
                        )
                    
                    if 'chat_messages' in session_data:
                        st.session_state.chat_messages = session_data['chat_messages']
                        st.session_state.chat_agent = None  # agent must be re-initialized

                    if 'document_updates' in session_data:
                        doc_updates = {}
                        for doc_name, updates in session_data['document_updates'].items():
                            doc_updates[doc_name] = [
                                DocumentUpdate(
                                    document_name=u['document_name'],
                                    version=u['version'],
                                    content=u['content'],
                                    gap_addressed=u.get('gap_addressed', ''),
                                )
                                for u in updates
                            ]
                        st.session_state.document_updates = doc_updates

                    st.success(f"Loaded session: {selected_session}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading session: {str(e)}")

        st.divider()
        st.subheader("Navigation")
        # Show context and audience status in sidebar
        audience_label = AUDIENCE_LABELS.get(
            st.session_state.get("doc_audience", "new_employee"), "—"
        )
        if st.session_state.process_context and st.session_state.process_context.is_set():
            ctx = st.session_state.process_context
            ctx_lines = []
            if ctx.foundation_document:
                ctx_lines.append(f"📄 Foundation: **{ctx.foundation_document}**")
            if ctx.process_description:
                preview = ctx.process_description[:80]
                ctx_lines.append(f"💬 {preview}{'…' if len(ctx.process_description) > 80 else ''}")
            ctx_lines.append(f"👥 Audience: **{audience_label}**")
            st.info("\n\n".join(ctx_lines))
        else:
            st.caption("No process context set — go to **Group Documents** to add one.")
            st.caption(f"👥 Audience: **{audience_label}**")

        page = st.radio(
            "Select a step:",
            [
                "Group Documents",
                "Analyze",
                "Review Process Document",
                "Gap Analysis",
                "Chat",
                "Knowledge Chat",
            ]
        )

        st.subheader("Settings")
        api_status = "✅ Configured" if config.AZURE_OPENAI_API_KEY else "❌ Not configured"
        st.write(f"Azure OpenAI: {api_status}")

        return page


def render_group_documents_page():
    """Unified page: load files, define context, build clusters, run hierarchical analysis."""
    from src.pipeline import FolderScanner, ClusterBuilder, HierarchicalSummarizer

    st.header("🗂️ Group Documents")
    st.write(
        "Load your documents, describe the process they relate to, then group them into "
        "logical clusters before running analysis."
    )

    # ── Step 1: Load files ────────────────────────────────────────────────────
    st.subheader("Step 1 — Load Files")

    load_mode = st.radio(
        "How would you like to load files?",
        ["Scan a folder path", "Upload individual files"],
        horizontal=True,
        key="group_load_mode",
    )

    if load_mode == "Scan a folder path":
        default_paths = ", ".join(str(p) for p in config.DOCUMENTS_PATHS)
        raw_paths = st.text_input(
            "Document folder path(s) — comma-separated for multiple folders",
            value=default_paths,
            placeholder="e.g. C:/projects/cobol_source, C:/projects/business_docs",
            help="Set DOCUMENTS_PATH in your .env file to pre-fill this field.",
        )
        input_paths = [Path(p.strip()) for p in raw_paths.split(",") if p.strip()]
        missing = [str(p) for p in input_paths if not p.exists()]
        if missing:
            st.warning(f"Path(s) not found: {', '.join(missing)}")

        recursive = st.checkbox("Scan sub-folders recursively", value=True)

        if st.button("🔍 Scan Folder(s)", disabled=not input_paths or bool(missing)):
            scanner = FolderScanner()
            with st.spinner("Scanning…"):
                st.session_state.bulk_scanned = scanner.scan(input_paths, recursive=recursive)
                st.session_state.bulk_clusters = None
                st.session_state.bulk_cluster_summaries = None
                st.session_state.uploaded_files = []

    else:  # Upload individual files
        uploaded = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["txt", "py", "cob", "cbl", "cic", "cpy", "docx", "doc", "xlsx", "xlsm", "xlsb", "html"],
        )
        if uploaded:
            for uf in uploaded:
                if uf.name not in [getattr(f, "name", str(f)) for f in st.session_state.uploaded_files]:
                    st.session_state.uploaded_files.append(uf)

        if st.session_state.uploaded_files:
            st.write(f"**{len(st.session_state.uploaded_files)} file(s) loaded:**")
            for idx, file in enumerate(st.session_state.uploaded_files):
                col_name, col_rm = st.columns([5, 1])
                col_name.caption(f"📄 {getattr(file, 'name', str(file))}")
                if col_rm.button("🗑️", key=f"rm_upload_{idx}"):
                    st.session_state.uploaded_files.pop(idx)
                    st.rerun()
        else:
            st.info("No files uploaded yet.")

    # Show scan results summary
    scanned = st.session_state.bulk_scanned
    if scanned:
        st.success(f"Found {scanned.summary()}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("COBOL files", len(scanned.cobol))
        col2.metric("Source code", len(scanned.code))
        col3.metric("Word docs", len(scanned.word))
        col4.metric("Excel files", len(scanned.excel))
        with st.expander("Show all discovered files"):
            if scanned.cobol:
                st.write("**COBOL**")
                for p in scanned.cobol: st.caption(str(p))
            if scanned.code:
                st.write("**Source Code** (Python, SQL, JS, etc.)")
                for p in scanned.code: st.caption(str(p))
            if scanned.word:
                st.write("**Word**")
                for p in scanned.word: st.caption(str(p))
            if scanned.excel:
                st.write("**Excel**")
                for p in scanned.excel: st.caption(str(p))

    files_ready = (scanned and scanned.total_count > 0) or bool(st.session_state.uploaded_files)

    # ── Step 2: Define Context ────────────────────────────────────────────────
    if files_ready:
        st.divider()
        st.subheader("Step 2 — Define Context (Optional but Recommended)")
        st.write(
            "Providing context helps the LLM understand the business domain and produce "
            "more accurate clusters and document sections. You can reference a specific "
            "document by name and the system will read it automatically."
        )
        ctx = st.session_state.process_context
        if ctx and ctx.is_set():
            with st.expander("✅ Context is set — click to view or change", expanded=False):
                if ctx.foundation_document:
                    st.write(f"**Foundation document:** {ctx.foundation_document}")
                if ctx.process_description:
                    st.write("**Process description:**")
                    st.write(ctx.process_description)
                if ctx.additional_notes:
                    st.write("**Additional notes:**")
                    st.write(ctx.additional_notes)
                if st.button("✏️ Clear and re-set context", key="group_clear_ctx"):
                    st.session_state.process_context = None
                    st.rerun()
        else:
            with st.expander("Set context now", expanded=True):
                quick_ctx = st.text_area(
                    "Describe the process — optionally reference a document by name as a foundation",
                    placeholder=(
                        "e.g. 'Payroll processing system for a mid-size manufacturer on IBM z/OS. "
                        "Use PAYROLL_OVERVIEW.docx as the foundation document.'"
                    ),
                    height=100,
                    key="group_quick_ctx",
                )
                if st.button("🔍 Extract Context", key="group_save_ctx") and quick_ctx.strip():
                    # Build list of Path objects for document detection
                    if scanned:
                        available_paths = scanned.cobol + scanned.code + scanned.word + scanned.excel
                    else:
                        available_paths = [
                            f for f in st.session_state.uploaded_files if isinstance(f, Path)
                        ]

                    ctx_agent = ProcessContextAgent(available_paths)

                    with st.spinner("Pass 1 — Checking for reference document…"):
                        detected_doc = ctx_agent._identify_reference_doc(quick_ctx.strip())

                    doc_content = ""
                    if detected_doc:
                        with st.spinner(f"Reading {detected_doc}…"):
                            doc_content = ctx_agent._read_doc(detected_doc)

                    with st.spinner("Extracting context…"):
                        ctx_result = ctx_agent._extract_context(
                            quick_ctx.strip(), doc_content, detected_doc
                        )

                    st.session_state.process_context = ctx_result
                    if detected_doc:
                        st.info(f"📄 Reference document detected and read: **{detected_doc}**")
                    st.success("Context saved.")
                    st.rerun()

        # Audience selector — always visible so it can be changed independently
        st.write("**Document Audience**")
        st.caption("Controls the writing style of all generated documentation and chat responses.")
        audience_options = list(AUDIENCE_LABELS.keys())
        current_idx = audience_options.index(
            st.session_state.get("doc_audience", "new_employee")
        )
        st.selectbox(
            "Who will read the generated documentation?",
            options=audience_options,
            format_func=lambda k: AUDIENCE_LABELS[k],
            index=current_idx,
            key="doc_audience",
            label_visibility="collapsed",
        )

    # ── Step 3: Build clusters ────────────────────────────────────────────────
    if files_ready:
        st.divider()
        st.subheader("Step 3 — Build Dependency Clusters")
        st.write(
            "Files are grouped into logical subsystem clusters. COBOL files are clustered "
            "by their CALL/COPY dependency graph; all other files are grouped by subject matter "
            "using the LLM, informed by the context you defined above."
        )

        ctx_block = _get_context_block()

        if st.button("🧩 Build Clusters", disabled=st.session_state.bulk_clusters is not None):
            builder = ClusterBuilder()
            with st.spinner("Analysing dependencies and clustering documents…"):
                try:
                    if scanned:
                        cobol = scanned.cobol
                        word = scanned.word
                        excel = scanned.excel
                        code = scanned.code
                    else:
                        # Uploaded files — sort by extension into buckets
                        import tempfile, shutil
                        tmp_dir = Path(tempfile.mkdtemp())
                        cobol, word, excel, code = [], [], [], []
                        word_exts = {".doc", ".docx"}
                        excel_exts = {".xlsx", ".xlsm", ".xlsb", ".xls"}
                        cobol_exts = {".cob", ".cbl", ".cic", ".cpy"}
                        for uf in st.session_state.uploaded_files:
                            ext = Path(getattr(uf, "name", str(uf))).suffix.lower()
                            dest = tmp_dir / getattr(uf, "name", str(uf))
                            if hasattr(uf, "read"):
                                dest.write_bytes(uf.read())
                            else:
                                shutil.copy(str(uf), str(dest))
                            if ext in cobol_exts: cobol.append(dest)
                            elif ext in word_exts: word.append(dest)
                            elif ext in excel_exts: excel.append(dest)
                            else: code.append(dest)

                    st.session_state.bulk_clusters = builder.build_clusters(
                        cobol_files=cobol,
                        word_files=word,
                        excel_files=excel,
                        code_files=code,
                        context_block=ctx_block,
                    )
                    st.session_state.bulk_cluster_summaries = None
                except Exception as e:
                    st.error(f"Clustering failed: {e}")

        if st.session_state.bulk_clusters:
            clusters = st.session_state.bulk_clusters
            st.success(f"Built {len(clusters)} clusters")
            for cl in clusters:
                label = {
                    "cobol": "🖥️ Source Code",
                    "mixed": "🔗 Code + Docs",
                    "docs": "📄 Documents",
                }.get(cl.cluster_type, cl.cluster_type)
                with st.expander(f"{label}: {cl.cluster_name} ({cl.file_count} files)"):
                    if cl.entry_point:
                        st.write(f"**Entry point:** {cl.entry_point}")
                    if cl.cobol_files:
                        st.write(f"**Source files ({len(cl.cobol_files)}):** " +
                                 ", ".join(p.name for p in cl.cobol_files[:20]) +
                                 ("…" if len(cl.cobol_files) > 20 else ""))
                    if cl.doc_files:
                        st.write(f"**Docs ({len(cl.doc_files)}):** " +
                                 ", ".join(p.name for p in cl.doc_files[:10]) +
                                 ("…" if len(cl.doc_files) > 10 else ""))
                    if cl.shared_doc_files:
                        st.write(f"**🌐 Shared reference docs ({len(cl.shared_doc_files)}):** " +
                                 ", ".join(p.name for p in cl.shared_doc_files[:10]) +
                                 ("…" if len(cl.shared_doc_files) > 10 else ""))

            if st.button("🔄 Re-cluster (clear and rebuild)"):
                st.session_state.bulk_clusters = None
                st.session_state.bulk_cluster_summaries = None
                st.rerun()

    # ── Step 4: Summarise clusters ────────────────────────────────────────────
    if st.session_state.bulk_clusters:
        st.divider()
        st.subheader("Step 4 — Summarize Clusters")
        clusters = st.session_state.bulk_clusters
        n = len(clusters)
        st.write(
            f"Each of the {n} clusters will be summarized separately, then synthesized "
            "into a single process document. "
            "This makes ~{} LLM calls — expect several minutes for large sets.".format(
                sum(len(cl.cobol_files) + len(cl.doc_files) + 1 for cl in clusters)
            )
        )

        if st.session_state.bulk_cluster_summaries is None:
            if st.button("▶️ Run Hierarchical Analysis"):
                summarizer = HierarchicalSummarizer()
                progress_bar = st.progress(0, text="Starting…")
                status_text = st.empty()

                def on_progress(name, idx, total):
                    frac = idx / total
                    progress_bar.progress(frac, text=f"Summarizing cluster {idx}/{total}")
                    status_text.write(f"Processing: **{name}**")

                ctx_block = _get_context_block()
                try:
                    summaries = summarizer.summarize_all(
                        clusters,
                        progress_callback=on_progress,
                        context_block=ctx_block,
                    )
                    st.session_state.bulk_cluster_summaries = summaries
                    st.session_state.analyses = [
                        AnalysisResult(
                            document_name=cs.cluster_name,
                            summary=cs.summary,
                            key_processes=cs.key_processes,
                            systems_mentioned=cs.systems_mentioned,
                            technical_details=[f"{cs.file_count} files in cluster"],
                        )
                        for cs in summaries
                    ]
                    progress_bar.progress(1.0, text="Done!")
                    status_text.empty()
                    try:
                        st.session_state.storage.save_analyses(
                            _ensure_session(),
                            st.session_state.analyses,
                        )
                        st.success(f"Summarized {len(summaries)} clusters. Saved to session: **{st.session_state.current_session}**")
                    except Exception as e:
                        st.success(f"Summarized {len(summaries)} clusters.")
                        st.warning(f"Could not save analyses: {e}")
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
        else:
            st.success(f"{len(st.session_state.bulk_cluster_summaries)} cluster summaries ready.")
            with st.expander("Preview cluster summaries"):
                for cs in st.session_state.bulk_cluster_summaries:
                    st.write(f"**{cs.cluster_name}** ({cs.file_count} files)")
                    st.write(cs.summary)
                    st.divider()

    # ── Step 5: Build process document ───────────────────────────────────────
    if st.session_state.bulk_cluster_summaries:
        st.divider()
        st.subheader("Step 5 — Build Process Document")
        st.write(
            "Synthesize the cluster summaries into one integrated process document. "
            "This is the same document you'll see in the Review Process Document page."
        )

        if st.button("📝 Build Process Document"):
            builder = ProcessDocumentBuilder()
            section_progress = st.progress(0, text="Starting…")
            section_status = st.empty()

            def on_section(label, idx, total):
                section_progress.progress(idx / total, text=f"Writing section {idx}/{total}")
                section_status.write(f"Generating: **{label}**")

            ctx_block = _get_context_block()
            try:
                process_doc = builder.build_from_cluster_summaries(
                    st.session_state.bulk_cluster_summaries,
                    progress_callback=on_section,
                    context_block=ctx_block,
                )
                section_progress.progress(1.0, text="Done!")
                section_status.empty()
                st.session_state.process_document = process_doc

                try:
                    st.session_state.storage.save_process_document(
                        _ensure_session(),
                        process_doc,
                        version="v1",
                    )
                    st.success(f"Process document built and saved to session: **{st.session_state.current_session}**. Navigate to **Review Process Document** to continue.")
                except Exception as save_err:
                    st.success("Process document built! Navigate to **Review Process Document** to continue.")
                    st.warning(f"Could not save: {save_err}")
                st.balloons()
            except Exception as e:
                st.error(f"Process document build failed: {e}")

        if st.session_state.process_document:
            st.info(
                "✅ Process document is ready. Use the sidebar to navigate to "
                "**Review Process Document → Gap Analysis → Questions → …**"
            )


def render_analyze_page():
    """Render document analysis page."""
    st.header("🔍 Analyze Documents")

    # If bulk mode was used, analyses are already populated — skip the upload-based flow
    if st.session_state.bulk_cluster_summaries and not st.session_state.uploaded_files:
        st.success(
            f"Analysis complete via Bulk Load — "
            f"{len(st.session_state.bulk_cluster_summaries)} clusters summarized."
        )
        st.info("Navigate to **Review Process Document** to continue the pipeline.")
        if st.session_state.analyses:
            st.divider()
            st.subheader("Cluster Summaries")
            for analysis in st.session_state.analyses:
                with st.expander(f"📋 {analysis.document_name}"):
                    st.write(analysis.summary)
        return

    if not st.session_state.uploaded_files:
        st.warning("Please upload documents first (or use **Bulk Load** for large document sets)")
        return

    if st.button("Start Analysis", type="primary"):
        with st.spinner("Analyzing documents..."):
            st.session_state.analyses = []
            progress_bar = st.progress(0)

            analyzer = DocumentAnalyzer()

            for idx, file in enumerate(st.session_state.uploaded_files):
                try:
                    # Get file path
                    if isinstance(file, Path):
                        file_path = file
                    else:
                        # For uploaded files, we need to save temporarily
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                            tmp.write(file.read())
                            file_path = Path(tmp.name)

                    # Validate and load
                    is_valid, error = validate_file(file_path)
                    if not is_valid:
                        st.error(f"Error with {file_path.name}: {error}")
                        continue

                    # Load document
                    loader = get_loader(file_path)
                    doc = loader.load()

                    # Analyze
                    analysis = analyzer.analyze_document(doc)
                    st.session_state.analyses.append(analysis)

                    progress_bar.progress((idx + 1) / len(st.session_state.uploaded_files))

                except Exception as e:
                    st.error(f"Error analyzing {getattr(file, 'name', 'file')}: {str(e)}")

            st.success(f"Analysis complete! {len(st.session_state.analyses)} documents analyzed")

            try:
                st.session_state.storage.save_analyses(
                    _ensure_session(),
                    st.session_state.analyses
                )
                st.info(f"✅ Saved to session: **{st.session_state.current_session}**")
            except Exception as e:
                st.error(f"Error saving analysis: {str(e)}")

    # Display analysis results
    if st.session_state.analyses:
        st.divider()
        st.subheader("Analysis Results")

        for analysis in st.session_state.analyses:
            with st.expander(f"📋 {analysis.document_name}"):
                st.write("**Summary:**")
                st.write(analysis.summary)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Key Processes:**")
                    for process in analysis.key_processes:
                        st.write(f"- {process}")

                with col2:
                    st.write("**Systems:**")
                    for system in analysis.systems_mentioned:
                        st.write(f"- {system}")

                with col3:
                    st.write("**Technical Details:**")
                    for detail in analysis.technical_details:
                        st.write(f"- {detail}")


def render_process_document_page():
    """Render process document review page."""
    st.header("📖 Process Document")

    if not st.session_state.analyses:
        st.warning("Please complete analysis first")
        return

    if st.button("Generate Process Document", type="primary"):
        builder = ProcessDocumentBuilder()
        section_progress = st.progress(0, text="Starting…")
        section_status = st.empty()

        def on_pd_section(label, idx, total):
            section_progress.progress(idx / total, text=f"Writing section {idx}/{total}")
            section_status.write(f"Generating: **{label}**")

        ctx_block = _get_context_block()
        try:
            st.session_state.process_document = builder.build_process_document(
                st.session_state.analyses,
                progress_callback=on_pd_section,
                context_block=ctx_block,
            )
            section_progress.progress(1.0, text="Done!")
            section_status.empty()

            try:
                st.session_state.storage.save_process_document(
                    _ensure_session(),
                    st.session_state.process_document,
                    version="v1"
                )
                st.success(f"✅ Process document generated and saved to session: **{st.session_state.current_session}**")
            except Exception as e:
                st.success("✅ Process document generated!")
                st.warning(f"Could not save: {e}")
        except Exception as e:
            st.error(f"Error generating process document: {str(e)}")

    if st.session_state.process_document:
        doc = st.session_state.process_document

        st.subheader("Overview")
        st.write(doc.overview)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Integrated Processes")
            st.write(doc.integrated_processes)

            st.subheader("Data Flow")
            st.write(doc.data_flow)

        with col2:
            st.subheader("Dependencies")
            st.write(doc.dependencies)

            st.subheader("Decision Points")
            st.write(doc.decision_points)

        st.divider()
        st.subheader("Systems & Components")
        st.write(doc.systems_and_components)

        if doc.appendix:
            st.divider()
            st.subheader("Appendix")
            st.write(doc.appendix)

        if doc.process_flow_diagram:
            st.divider()
            st.subheader("📊 Process Flow Diagram")
            tab_interactive, tab_png = st.tabs(["Interactive (Mermaid)", "Static PNG"])
            with tab_interactive:
                _render_mermaid(doc.process_flow_diagram, height=560)
                with st.expander("View / copy Mermaid source"):
                    st.code(doc.process_flow_diagram, language="text")
            with tab_png:
                with st.spinner("Rendering PNG…"):
                    png_bytes = _mermaid_to_png(doc.process_flow_diagram)
                if png_bytes:
                    st.image(png_bytes, use_container_width=True)
                    st.download_button(
                        "📥 Download diagram PNG",
                        data=png_bytes,
                        file_name="process_flow.png",
                        mime="image/png",
                    )
                else:
                    st.info("PNG rendering unavailable — check network access to mermaid.ink. The Word export will include the Mermaid source as a fallback.")

        # Export buttons
        st.divider()
        appendix_section = f"\n\n## Appendix\n{doc.appendix}" if doc.appendix else ""
        diagram_section = (
            f"\n\n## Process Flow Diagram\n```mermaid\n{doc.process_flow_diagram}\n```"
            if doc.process_flow_diagram else ""
        )
        markdown = f"""# Process Document

## Overview
{doc.overview}

## Integrated Processes
{doc.integrated_processes}

## Dependencies
{doc.dependencies}

## Data Flow
{doc.data_flow}

## Decision Points
{doc.decision_points}

## Systems & Components
{doc.systems_and_components}{appendix_section}{diagram_section}
"""
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=markdown,
                file_name="process_document.md",
                mime="text/markdown",
            )
        with col2:
            st.download_button(
                label="📄 Download as Word",
                data=_generate_word_doc(doc),
                file_name="process_document.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


def _map_gaps_to_clusters(gap_descriptions: list[str], analyses: list) -> dict[str, str]:
    """Map each gap description to the best-matching cluster/analysis name via word overlap.

    Returns a dict of {gap_description: cluster_name}. Falls back to the first analysis
    when there is no meaningful overlap (so the caller always gets a name).
    """
    import re
    if not analyses or not gap_descriptions:
        return {}
    result: dict[str, str] = {}
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "or", "and", "of",
                 "in", "to", "for", "with", "not", "this", "that", "has", "have",
                 "it", "its", "be", "by", "at", "on", "as", "no", "gaps", "gap",
                 "missing", "undefined", "incomplete", "each", "identified"}
    for gap in gap_descriptions:
        gap_words = set(re.findall(r'\w+', gap.lower())) - stopwords
        best_name = analyses[0].document_name
        best_score = -1
        for a in analyses:
            candidate = (
                a.summary + " "
                + " ".join(a.key_processes)
                + " " + " ".join(a.systems_mentioned)
            ).lower()
            overlap = len(gap_words & (set(re.findall(r'\w+', candidate)) - stopwords))
            if overlap > best_score:
                best_score, best_name = overlap, a.document_name
        result[gap] = best_name
    return result


def render_gap_analysis_page():
    """Render gap analysis page."""
    st.header("⚠️ Gap Analysis")

    if not st.session_state.process_document:
        st.warning("Please generate process document first")
        return

    if st.button("Analyze Gaps", type="primary"):
        with st.spinner("Analyzing gaps and missing information..."):
            gap_analyzer = GapAnalyzer()
            st.session_state.gap_analysis = gap_analyzer.analyze_gaps(
                st.session_state.process_document,
                context_block=_get_context_block(),
            )
            
            try:
                st.session_state.storage.save_gap_analysis(
                    _ensure_session(),
                    st.session_state.gap_analysis,
                    version="v1"
                )
                st.success(f"✅ Gap analysis complete and saved to session: **{st.session_state.current_session}**")
            except Exception as e:
                st.error(f"Error saving gap analysis: {str(e)}")

    if st.session_state.gap_analysis:
        gaps = st.session_state.gap_analysis

        # Build gap-to-cluster map once for the whole page
        all_gap_descriptions = (
            gaps.missing_steps + gaps.undefined_dependencies
            + gaps.incomplete_transformations + gaps.missing_integrations
            + gaps.error_handling_gaps + gaps.security_gaps + gaps.resource_gaps
        )
        gap_cluster_map = _map_gaps_to_clusters(
            all_gap_descriptions, st.session_state.analyses
        )

        def _gap_item(item: str) -> None:
            cluster = gap_cluster_map.get(item, "")
            badge = f" — *{cluster}*" if cluster else ""
            st.markdown(f"- {item}{badge}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("❌ Missing Steps")
            for item in gaps.missing_steps:
                _gap_item(item)

        with col2:
            st.subheader("❓ Undefined Dependencies")
            for item in gaps.undefined_dependencies:
                _gap_item(item)

        with col3:
            st.subheader("🔄 Incomplete Transformations")
            for item in gaps.incomplete_transformations:
                _gap_item(item)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔗 Missing Integrations")
            for item in gaps.missing_integrations:
                _gap_item(item)

        with col2:
            st.subheader("🚨 Error Handling Gaps")
            for item in gaps.error_handling_gaps:
                _gap_item(item)

        with col3:
            st.subheader("🔒 Security Gaps")
            for item in gaps.security_gaps:
                _gap_item(item)

        st.subheader("🧑‍💼 Resource Gaps")
        for item in gaps.resource_gaps:
            _gap_item(item)


def render_chat_page():
    """Conversational gap-filling chat with the AI agent."""
    st.header("💬 Gap-Filling Chat")

    if not st.session_state.analyses:
        st.warning("Please analyze documents first.")
        return
    if not st.session_state.process_document:
        st.warning("Please generate the process document first.")
        return
    if not st.session_state.gap_analysis:
        st.warning("Please run gap analysis first.")
        return

    # Loaded session: messages exist but no live agent
    if st.session_state.chat_agent is None and st.session_state.chat_messages:
        st.info("Previous chat session loaded. Start a new session to continue the conversation.")
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if st.button("▶️ Start New Chat Session"):
            st.session_state.chat_messages = []
            st.session_state.document_updates = {}
            st.rerun()
        return

    # Initialize agent on first visit
    if st.session_state.chat_agent is None:
        with st.spinner("Initializing chat agent..."):
            # Load knowledge base for gap-to-source linking if it has been built
            _kb = None
            _session = st.session_state.current_session
            if _session:
                _kb_dir = Path("./analysis_sessions") / _session / "knowledge_base"
                _kb_candidate = KnowledgeBase(_kb_dir)
                if _kb_candidate.is_built():
                    _kb = _kb_candidate

            _audience = st.session_state.get("doc_audience", "new_employee")
            agent = ConversationalAgent(
                analyses=st.session_state.analyses,
                process_document=st.session_state.process_document,
                gap_analysis=st.session_state.gap_analysis,
                knowledge_base=_kb,
                audience_note=AUDIENCE_NOTES.get(_audience, ""),
            )
            opening = agent.get_opening_message()
            st.session_state.chat_agent = agent
            st.session_state.chat_messages = [{"role": "assistant", "content": opening}]
            st.session_state.document_updates = {}
        st.rerun()

    agent: ConversationalAgent = st.session_state.chat_agent
    total_gaps = len(agent.gap_queue)
    resolved_gaps = sum(1 for g in agent.gap_queue if g.resolved)

    # ── Gap tracker (scrollable box, sits above the chat) ────────────────────
    if total_gaps > 0:
        st.subheader(f"Gaps  ({resolved_gaps}/{total_gaps} resolved)")

        # Map each gap to its most relevant cluster (word-overlap, no LLM call)
        gap_cluster_map = _map_gaps_to_clusters(
            [g.description for g in agent.gap_queue],
            st.session_state.analyses,
        )

        by_category: dict = defaultdict(list)
        for g in agent.gap_queue:
            by_category[g.category].append(g)

        import html as _html
        gap_rows = []
        for category, items in by_category.items():
            gap_rows.append(
                f"<div style='font-size:0.75rem;font-weight:600;color:#888;"
                f"text-transform:uppercase;margin-top:8px;padding:2px 0'>"
                f"{_html.escape(category.replace('_', ' '))}</div>"
            )
            for g in items:
                cluster = gap_cluster_map.get(g.description, "")
                if g.resolved:
                    icon = "✅"
                    row_style = "color:#4caf50;border-left:3px solid #4caf50;"
                    tip = (
                        f" title='{_html.escape(g.resolution[:120])}'"
                        if g.resolution else ""
                    )
                    gap_rows.append(
                        f"<div{tip} style='font-size:0.82rem;{row_style}"
                        f"padding:4px 0 4px 8px;margin:3px 0;line-height:1.4'>"
                        f"{icon}&nbsp;{_html.escape(g.description)}</div>"
                    )
                else:
                    icon = "⬜"
                    row_style = "color:#333;border-left:3px solid #ddd;"
                    gap_rows.append(
                        f"<div style='font-size:0.82rem;{row_style}"
                        f"padding:4px 0 2px 8px;margin:3px 0;line-height:1.4'>"
                        f"{icon}&nbsp;{_html.escape(g.description)}</div>"
                    )
                    # Show cluster tag and source files under unresolved gaps
                    meta_parts = []
                    if cluster:
                        meta_parts.append(f"🏷️ {_html.escape(cluster)}")
                    if g.related_sources:
                        src_names = ", ".join(
                            _html.escape(s['file']) for s in g.related_sources[:3]
                        )
                        meta_parts.append(f"📎 {src_names}")
                    if meta_parts:
                        gap_rows.append(
                            f"<div style='font-size:0.72rem;color:#888;"
                            f"padding:0 0 4px 22px;line-height:1.3'>"
                            f"{'&nbsp;&nbsp;·&nbsp;&nbsp;'.join(meta_parts)}</div>"
                        )

        html_content = "\n".join(gap_rows)
        st.markdown(
            f"<div style='max-height:260px;overflow-y:auto;padding:6px 10px;"
            f"border:1px solid #e0e0e0;border-radius:6px;background:#fafafa;"
            f"margin-bottom:16px'>"
            f"{html_content}</div>",
            unsafe_allow_html=True,
        )

    _render_chat_column(agent)


def _render_chat_column(agent) -> None:
    """Reset button, chat history, document updates sidebar, and chat input."""
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_agent = None
        st.session_state.chat_messages = []
        st.session_state.document_updates = {}
        st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── Document updates sidebar ──────────────────────────────────────────────
    if st.session_state.document_updates:
        with st.sidebar:
            st.divider()
            st.subheader("📄 Updated Documents")
            for doc_name, updates in st.session_state.document_updates.items():
                st.write(f"**{doc_name}** — {len(updates)} update(s)")
                for upd in updates:
                    label = f"v{upd.version}: {upd.gap_addressed[:35]}..."
                    with st.expander(label):
                        st.text_area(
                            "Content",
                            upd.content,
                            height=200,
                            key=f"upd_{doc_name}_{upd.version}",
                        )
                        st.download_button(
                            "📥 Download",
                            data=upd.content,
                            file_name=f"{doc_name}_v{upd.version}.md",
                            mime="text/markdown",
                            key=f"dl_{doc_name}_{upd.version}",
                        )

    # ── Input / completion ────────────────────────────────────────────────────
    if len(agent.remaining_gaps) == 0:
        st.success("✅ All gaps have been addressed! Download updated documents from the sidebar.")

        if st.session_state.document_updates:
            if st.button("💾 Save All Updates to Session"):
                session = _ensure_session()
                errors = []
                for doc_name, updates in st.session_state.document_updates.items():
                    for upd in updates:
                        try:
                            st.session_state.storage.save_document_update(
                                session,
                                upd.document_name,
                                upd.version,
                                upd.content,
                                upd.gap_addressed,
                            )
                        except Exception as e:
                            errors.append(f"{doc_name}: {e}")
                if errors:
                    st.error("Some saves failed: " + "; ".join(errors))
                else:
                    st.success(f"All document updates saved to session: **{session}**")
    else:
        if user_input := st.chat_input("Type your response..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                response = agent.chat(user_input)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response.message}
            )

            for update in response.document_updates:
                st.session_state.document_updates.setdefault(update.document_name, [])
                st.session_state.document_updates[update.document_name].append(update)
                try:
                    st.session_state.storage.save_document_update(
                        _ensure_session(),
                        update.document_name,
                        update.version,
                        update.content,
                        update.gap_addressed,
                    )
                except Exception:
                    pass

            try:
                st.session_state.storage.save_chat_messages(
                    _ensure_session(),
                    st.session_state.chat_messages,
                )
            except Exception:
                pass

            st.rerun()



def _add_content_to_docx(docx, content: str) -> None:
    """Convert LLM markdown-style output into Word paragraphs."""
    import re
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            docx.add_heading(stripped[4:].strip(), level=3)
        elif stripped.startswith("## "):
            docx.add_heading(stripped[3:].strip(), level=2)
        elif stripped.startswith("# "):
            docx.add_heading(stripped[2:].strip(), level=2)
        elif re.match(r"^[-*•]\s+", stripped):
            text = re.sub(r"^[-*•]\s+", "", stripped)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            docx.add_paragraph(text, style="List Bullet")
        elif re.match(r"^\d+[.)]\s+", stripped):
            text = re.sub(r"^\d+[.)]\s+", "", stripped)
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            docx.add_paragraph(text, style="List Number")
        else:
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            docx.add_paragraph(text)


def _mermaid_to_png(mermaid_code: str) -> bytes | None:
    """Fetch a PNG render of Mermaid syntax from the mermaid.ink public API."""
    import base64
    import urllib.request
    clean = mermaid_code.strip()
    for fence in ("```mermaid", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()
    try:
        encoded = base64.urlsafe_b64encode(clean.encode()).decode()
        url = f"https://mermaid.ink/img/{encoded}?bgColor=white"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return resp.read()
    except Exception:
        pass
    return None


def _generate_word_doc(doc) -> bytes:
    """Build a Word document from a ProcessDocument and return raw bytes."""
    import io
    from docx import Document as DocxDocument
    from docx.shared import Inches, Pt

    d = DocxDocument()
    d.add_heading("Process Document", 0)

    sections = [
        ("Overview", doc.overview),
        ("Integrated Processes", doc.integrated_processes),
        ("Dependencies", doc.dependencies),
        ("Data Flow", doc.data_flow),
        ("Decision Points", doc.decision_points),
        ("Systems & Components", doc.systems_and_components),
    ]
    if doc.appendix:
        sections.append(("Appendix", doc.appendix))

    for title, content in sections:
        d.add_heading(title, level=1)
        _add_content_to_docx(d, content)

    if doc.process_flow_diagram:
        d.add_heading("Process Flow Diagram", level=1)
        png = _mermaid_to_png(doc.process_flow_diagram)
        if png:
            d.add_picture(io.BytesIO(png), width=Inches(6.0))
        else:
            # Offline fallback: embed Mermaid source as code block
            d.add_paragraph(
                "To render this diagram, paste the code below at https://mermaid.live"
            )
            code_para = d.add_paragraph(doc.process_flow_diagram)
            for run in code_para.runs:
                run.font.name = "Courier New"
                run.font.size = Pt(9)

    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _render_mermaid(diagram_code: str, height: int = 500) -> None:
    """Render a Mermaid diagram using the Mermaid.js CDN via st.components."""
    # Strip markdown code fences if the LLM wrapped the output
    clean = diagram_code.strip()
    for fence in ("```mermaid", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{startOnLoad:true, theme:'default', securityLevel:'loose'}});</script>
  <style>
    body {{ margin: 0; padding: 8px; background: #fff; }}
    .mermaid {{ width: 100%; overflow: auto; }}
  </style>
</head>
<body>
  <div class="mermaid">{clean}</div>
</body>
</html>"""
    st.components.v1.html(html, height=height, scrolling=True)


def render_knowledge_chat_page() -> None:
    """RAG-powered Q&A over the original source documents."""
    st.header("🔎 Knowledge Chat")
    st.write(
        "Ask questions about the process — answers are pulled directly from the "
        "original source files (Word, Excel, code). "
        "This is separate from the document-generation pipeline."
    )

    # Determine which files are available to index
    scanned = st.session_state.get("bulk_scanned")
    all_files = []
    if scanned:
        all_files = scanned.cobol + scanned.code + scanned.word + scanned.excel
    elif st.session_state.uploaded_files:
        all_files = [
            f for f in st.session_state.uploaded_files if isinstance(f, Path)
        ]

    if not all_files:
        st.warning(
            "No documents loaded yet. Go to **Bulk Load** or **Upload Documents** first, "
            "then come back here to build the knowledge base."
        )
        return

    # ── Knowledge base status ─────────────────────────────────────────
    session = st.session_state.current_session
    kb_dir = (
        Path("./analysis_sessions") / session / "knowledge_base"
        if session
        else Path("./knowledge_base_temp")
    )
    kb = KnowledgeBase(kb_dir)

    if not kb.is_built():
        st.info(
            f"The knowledge base has not been built yet. "
            f"{len(all_files)} files are ready to index."
        )
        if st.button("🏗️ Build Knowledge Base", type="primary"):
            progress_bar = st.progress(0, text="Starting…")
            status_text = st.empty()

            def on_kb_progress(msg, current, total):
                frac = current / total if total else 0
                progress_bar.progress(frac, text=msg)
                status_text.write(msg)

            try:
                n_chunks = kb.build(all_files, progress_callback=on_kb_progress)

                # Also index the generated process document as tier-1 (highest priority)
                proc_doc = st.session_state.get("process_document")
                extra_chunks = 0
                if proc_doc:
                    status_text.write("Indexing generated process document…")
                    extra_chunks = kb.index_process_document(proc_doc)

                progress_bar.progress(1.0, text="Done!")
                status_text.empty()
                extra_msg = f" + {extra_chunks} process-doc chunks" if extra_chunks else ""
                st.success(
                    f"Knowledge base built: {n_chunks} chunks from {len(all_files)} files{extra_msg}."
                )
                # Initialise agent immediately
                _audience = st.session_state.get("doc_audience", "new_employee")
                st.session_state.rag_agent = RAGAgent(
                    kb, audience_note=AUDIENCE_NOTES.get(_audience, "")
                )
                st.session_state.rag_messages = []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to build knowledge base: {e}")
        return

    # ── Initialise agent if not already loaded ────────────────────────
    if st.session_state.rag_agent is None:
        # Re-index process document on load if available and not yet in the KB
        proc_doc = st.session_state.get("process_document")
        if proc_doc:
            kb.index_process_document(proc_doc)
        _audience = st.session_state.get("doc_audience", "new_employee")
        st.session_state.rag_agent = RAGAgent(
            kb, audience_note=AUDIENCE_NOTES.get(_audience, "")
        )
        stats = kb.get_stats()
        st.success(
            f"Knowledge base loaded: {stats['chunks']} chunks from {stats['files']} files."
        )

    agent: RAGAgent = st.session_state.rag_agent
    stats = kb.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Source files", stats["files"])
    col2.metric("Index chunks", stats["chunks"])
    col3.metric("Messages", sum(1 for m in st.session_state.rag_messages if m["role"] == "user"))

    if st.button("🗑️ Clear Conversation"):
        agent.reset()
        st.session_state.rag_messages = []
        st.rerun()

    if st.button("🔄 Rebuild Knowledge Base"):
        import shutil
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
        st.session_state.rag_agent = None
        st.session_state.rag_messages = []
        st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("enriched_query") and msg["enriched_query"] != msg.get("raw_query"):
                st.caption(f"🔍 Searched for: *{msg['enriched_query']}*")
            if msg.get("sources"):
                with st.expander("📄 Source excerpts used"):
                    for src in msg["sources"]:
                        score_pct = int(src["score"] * 100)
                        st.markdown(
                            f"**{src['source_file']}** — relevance {score_pct}%\n\n"
                            f"> {src['text'][:300]}{'…' if len(src['text']) > 300 else ''}"
                        )

    # ── Input ──────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask anything about the process or documents…"):
        st.session_state.rag_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Enriching query and searching knowledge base…"):
                try:
                    answer, sources, enriched_query = agent.chat(user_input)
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                    sources = []
                    enriched_query = user_input
            st.write(answer)
            if enriched_query and enriched_query != user_input:
                st.caption(f"🔍 Searched for: *{enriched_query}*")
            if sources:
                with st.expander("📄 Source excerpts used"):
                    for src in sources:
                        score_pct = int(src["score"] * 100)
                        st.markdown(
                            f"**{src['source_file']}** — relevance {score_pct}%\n\n"
                            f"> {src['text'][:300]}{'…' if len(src['text']) > 300 else ''}"
                        )

        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "enriched_query": enriched_query,
            "raw_query": user_input,
        })
        st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Document Analysis System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 0rem;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session()

    page = render_sidebar()

    # Main content
    if page == "Group Documents":
        render_group_documents_page()
    elif page == "Analyze":
        render_analyze_page()
    elif page == "Review Process Document":
        render_process_document_page()
    elif page == "Gap Analysis":
        render_gap_analysis_page()
    elif page == "Chat":
        render_chat_page()
    elif page == "Knowledge Chat":
        render_knowledge_chat_page()

    # Footer
    st.divider()
    _, col2, _ = st.columns(3)
    with col2:
        st.caption("Document Analysis System v2.0 - With Session Management & Document Refinement")


if __name__ == "__main__":
    main()
