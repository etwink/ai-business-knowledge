"""Main Streamlit application for Document Analysis System."""

import streamlit as st
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

# Inject truststore into SSL context for better certificate handling (especially in enterprise environments)
import truststore
truststore.inject_into_ssl()

from src.document_loaders import get_loader, DocumentContent
from src.analyzers import (
    DocumentAnalyzer,
    ProcessDocumentBuilder,
    GapAnalyzer,
    ClarificationQuestionGenerator,
    AnalysisResult,
    ProcessDocument,
    GapAnalysis
)
from src.utils import get_supported_documents, validate_file
from src.storage import AnalysisStorage
from src.pipeline import ConversationalAgent, DocumentUpdate, ProcessContextAgent, ProcessContext
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
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
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
    if 'context_agent' not in st.session_state:
        st.session_state.context_agent = None
    if 'context_messages' not in st.session_state:
        st.session_state.context_messages = []
    # Chat agent state
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'document_updates' not in st.session_state:
        st.session_state.document_updates = {}  # doc_name -> list[DocumentUpdate]


def _ensure_session() -> str:
    """Return current session name, auto-creating a timestamped one if not set."""
    if not st.session_state.current_session:
        st.session_state.current_session = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return st.session_state.current_session


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
        # Show context status in sidebar
        if st.session_state.process_context and st.session_state.process_context.is_set():
            ctx = st.session_state.process_context
            ctx_lines = []
            if ctx.foundation_document:
                ctx_lines.append(f"📄 Foundation: **{ctx.foundation_document}**")
            if ctx.process_description:
                preview = ctx.process_description[:80]
                ctx_lines.append(f"💬 {preview}{'…' if len(ctx.process_description) > 80 else ''}")
            st.info("\n\n".join(ctx_lines))
        else:
            st.caption("No process context set — go to **Define Context** to add one.")

        page = st.radio(
            "Select a step:",
            [
                "Upload Documents",
                "Bulk Load (130+ Files)",
                "Define Context",
                "Analyze",
                "Review Process Document",
                "Gap Analysis",
                "Chat",
            ]
        )

        st.subheader("Settings")
        api_status = "✅ Configured" if config.AZURE_OPENAI_API_KEY else "❌ Not configured"
        st.write(f"Azure OpenAI: {api_status}")

        return page


def render_upload_page():
    """Render document upload page."""
    st.header("📤 Upload Documents")

    st.write("Upload documents for analysis (COBOL, Word, Excel, HTML, Text, PDF)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'py', 'cob', 'cbl', 'cic', 'cpy', 'docx', 'xlsx', 'html']
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f.name for f in st.session_state.uploaded_files]:
                    st.session_state.uploaded_files.append(uploaded_file)
                    st.success(f"Added: {uploaded_file.name}")

    with col2:
        st.subheader("Or Load from Folder")
        sample_docs = get_supported_documents(config.SAMPLE_DOCS_DIR)
        for doc_path in sample_docs:
            if st.button(f"Load: {doc_path.name}", key=f"load_{doc_path}"):
                # Read the file
                with open(doc_path, 'rb') as f:
                    content = f.read()
                # Create a mock uploaded file object
                class MockUploadedFile:
                    def __init__(self, name, content):
                        self.name = name
                        self.content = content
                # Add to session
                if not any(f.name == doc_path.name for f in st.session_state.uploaded_files):
                    st.session_state.uploaded_files.append(doc_path)

    st.divider()
    st.subheader("Uploaded Files")

    if st.session_state.uploaded_files:
        for idx, file in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"📄 {getattr(file, 'name', str(file))}")
            with col2:
                if st.button("🗑️", key=f"remove_{idx}"):
                    st.session_state.uploaded_files.pop(idx)
                    st.rerun()
    else:
        st.info("No files uploaded yet")


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

        ctx_block = (
            st.session_state.process_context.to_prompt_block()
            if st.session_state.process_context else ""
        )
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

        # Export button
        if st.button("📥 Export as Markdown"):
            appendix_section = f"\n\n## Appendix\n{doc.appendix}" if doc.appendix else ""
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
{doc.systems_and_components}{appendix_section}
"""
            st.download_button(
                label="Download Markdown",
                data=markdown,
                file_name="process_document.md",
                mime="text/markdown"
            )


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
                st.session_state.process_document
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

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("❌ Missing Steps")
            for item in gaps.missing_steps:
                st.write(f"- {item}")

        with col2:
            st.subheader("❓ Undefined Dependencies")
            for item in gaps.undefined_dependencies:
                st.write(f"- {item}")

        with col3:
            st.subheader("🔄 Incomplete Transformations")
            for item in gaps.incomplete_transformations:
                st.write(f"- {item}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔗 Missing Integrations")
            for item in gaps.missing_integrations:
                st.write(f"- {item}")

        with col2:
            st.subheader("🚨 Error Handling Gaps")
            for item in gaps.error_handling_gaps:
                st.write(f"- {item}")

        with col3:
            st.subheader("🔒 Security Gaps")
            for item in gaps.security_gaps:
                st.write(f"- {item}")

        st.subheader("🧑‍💼 Resource Gaps")
        for item in gaps.resource_gaps:
            st.write(f"- {item}")


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
            agent = ConversationalAgent(
                analyses=st.session_state.analyses,
                process_document=st.session_state.process_document,
                gap_analysis=st.session_state.gap_analysis,
            )
            opening = agent.get_opening_message()
            st.session_state.chat_agent = agent
            st.session_state.chat_messages = [{"role": "assistant", "content": opening}]
            st.session_state.document_updates = {}
        st.rerun()

    agent: ConversationalAgent = st.session_state.chat_agent
    total_gaps = len(agent.gap_queue)
    resolved_gaps = sum(1 for g in agent.gap_queue if g.resolved)

    # ── Progress bar ──────────────────────────────────────────────────────────
    if total_gaps > 0:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.progress(
                resolved_gaps / total_gaps,
                text=f"Gaps resolved: {resolved_gaps} / {total_gaps}",
            )
        with col2:
            st.metric("Resolved", resolved_gaps)
        with col3:
            st.metric("Remaining", total_gaps - resolved_gaps)

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

            # Handle document updates
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

            # Auto-save chat messages after every turn
            try:
                st.session_state.storage.save_chat_messages(
                    _ensure_session(),
                    st.session_state.chat_messages,
                )
            except Exception:
                pass

            st.rerun()


def render_context_page():
    """Let the user identify a foundation document and describe the process before analysis."""
    st.header("🎯 Define Process Context")
    st.write(
        "Before analysis begins, tell the system what this process is about. "
        "You can identify a key document that should anchor the documentation, "
        "describe the overall purpose in your own words, or both. "
        "This context is injected into every LLM call so the output stays focused on the right goal."
    )

    # ── Gather available document names from whatever has been loaded ──────────
    available_docs: list[str] = []
    if st.session_state.bulk_scanned:
        scanned = st.session_state.bulk_scanned
        available_docs = (
            [p.name for p in scanned.cobol]
            + [p.name for p in scanned.code]
            + [p.name for p in scanned.word]
            + [p.name for p in scanned.excel]
        )
    elif st.session_state.uploaded_files:
        available_docs = [
            getattr(f, "name", str(f)) for f in st.session_state.uploaded_files
        ]

    # ── Show current context if already set ───────────────────────────────────
    if st.session_state.process_context and st.session_state.process_context.is_set():
        ctx = st.session_state.process_context
        st.success("✅ Process context is set.")
        with st.expander("Current context (click to view)", expanded=True):
            if ctx.foundation_document:
                st.write(f"**Foundation document:** {ctx.foundation_document}")
            if ctx.process_description:
                st.write("**Process description:**")
                st.write(ctx.process_description)
            if ctx.additional_notes:
                st.write("**Additional notes:**")
                st.write(ctx.additional_notes)
        if st.button("🔄 Reset Context"):
            st.session_state.process_context = None
            st.session_state.context_agent = None
            st.session_state.context_messages = []
            st.rerun()
        return

    # ── Initialize agent ──────────────────────────────────────────────────────
    if st.session_state.context_agent is None:
        if not available_docs:
            st.info(
                "No documents are loaded yet. Go to **Upload Documents** or "
                "**Bulk Load** first, then come back here to set context."
            )
            # Still allow free-text context even without documents
            available_docs = ["(no files loaded)"]

        with st.spinner("Initializing context agent…"):
            agent = ProcessContextAgent(available_docs)
            opening = agent.get_opening_message()
            st.session_state.context_agent = agent
            st.session_state.context_messages = [
                {"role": "assistant", "content": opening}
            ]
        st.rerun()

    agent: ProcessContextAgent = st.session_state.context_agent

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.context_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # ── If agent flagged context ready, show confirm button ───────────────────
    if agent.context_ready and agent.extracted_context:
        ctx = agent.extracted_context
        st.divider()
        st.subheader("Extracted context — please confirm")
        col1, col2 = st.columns(2)
        with col1:
            if ctx.foundation_document:
                st.write(f"**Foundation document:** {ctx.foundation_document}")
            else:
                st.write("**Foundation document:** *(none specified)*")
        with col2:
            if ctx.process_description:
                st.write("**Process description:**")
                st.write(ctx.process_description)

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✅ Confirm & Save Context", type="primary"):
                st.session_state.process_context = ctx
                st.success("Context saved! You can now proceed to Analyze.")
                st.rerun()
        with btn_col2:
            if st.button("✏️ Refine Further"):
                agent.context_ready = False
                st.rerun()
        return

    # ── Confirm button even if agent hasn't flagged ready ────────────────────
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.session_state.context_messages and len(st.session_state.context_messages) > 1:
            if st.button("✅ Done — Extract Context"):
                with st.spinner("Extracting context from conversation…"):
                    ctx = agent.force_extract()
                st.session_state.process_context = ctx
                st.success("Context saved!")
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Describe the process or identify a key document…"):
        st.session_state.context_messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking…"):
            response, extracted = agent.chat(user_input)
        st.session_state.context_messages.append({"role": "assistant", "content": response})
        if extracted:
            st.session_state.process_context = extracted
        st.rerun()


def render_bulk_load_page():
    """Bulk mode: scan a folder, build dependency-aware clusters, and run hierarchical summarization."""
    from src.pipeline import FolderScanner, ClusterBuilder, HierarchicalSummarizer

    st.header("📂 Bulk Load — Large Document Sets")
    st.write(
        "Use this page when you have 50+ files. Documents are grouped into logical "
        "subsystems before analysis so the final process document stays manageable."
    )

    # ── Step 1: Configure path ────────────────────────────────────────────────
    st.subheader("Step 1 — Configure Document Path")

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

    # ── Step 2: Scan ─────────────────────────────────────────────────────────
    st.subheader("Step 2 — Scan for Files")

    if st.button("🔍 Scan Folder(s)", disabled=not input_paths or bool(missing)):
        scanner = FolderScanner()
        with st.spinner("Scanning…"):
            st.session_state.bulk_scanned = scanner.scan(input_paths, recursive=recursive)
            st.session_state.bulk_clusters = None
            st.session_state.bulk_cluster_summaries = None

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
                for p in scanned.cobol:
                    st.caption(str(p))
            if scanned.code:
                st.write("**Source Code** (Python, SQL, JS, etc.)")
                for p in scanned.code:
                    st.caption(str(p))
            if scanned.word:
                st.write("**Word**")
                for p in scanned.word:
                    st.caption(str(p))
            if scanned.excel:
                st.write("**Excel**")
                for p in scanned.excel:
                    st.caption(str(p))

    # ── Step 3: Build clusters ────────────────────────────────────────────────
    if scanned and scanned.total_count > 0:
        st.subheader("Step 3 — Build Dependency Clusters")
        st.write(
            "COBOL files are grouped by their CALL/COPY dependency graph into subsystem clusters. "
            "Other source code files (Python, SQL, JS, etc.) are treated as business documents "
            "and grouped by subject matter using the LLM. "
            "Word and Excel files are first matched to source clusters by program-name mentions, "
            "then any remaining docs are also LLM-clustered by business domain."
        )

        if st.button("🧩 Build Clusters", disabled=st.session_state.bulk_clusters is not None):
            builder = ClusterBuilder()
            with st.spinner("Analysing dependencies and clustering documents…"):
                try:
                    st.session_state.bulk_clusters = builder.build_clusters(
                        cobol_files=scanned.cobol,
                        word_files=scanned.word,
                        excel_files=scanned.excel,
                        code_files=scanned.code,
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
        st.subheader("Step 4 — Summarize Clusters")
        clusters = st.session_state.bulk_clusters
        n = len(clusters)
        st.write(
            f"Each of the {n} clusters will be summarized separately using the LLM, "
            "then synthesized into a single process document. "
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

                ctx_block = (
                    st.session_state.process_context.to_prompt_block()
                    if st.session_state.process_context else ""
                )
                try:
                    summaries = summarizer.summarize_all(
                        clusters,
                        progress_callback=on_progress,
                        context_block=ctx_block,
                    )
                    st.session_state.bulk_cluster_summaries = summaries

                    # Convert to AnalysisResult for storage / downstream compatibility
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
            st.success(
                f"{len(st.session_state.bulk_cluster_summaries)} cluster summaries ready."
            )
            with st.expander("Preview cluster summaries"):
                for cs in st.session_state.bulk_cluster_summaries:
                    st.write(f"**{cs.cluster_name}** ({cs.file_count} files)")
                    st.write(cs.summary)
                    st.divider()

    # ── Step 5: Build process document ───────────────────────────────────────
    if st.session_state.bulk_cluster_summaries:
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

            ctx_block = (
                st.session_state.process_context.to_prompt_block()
                if st.session_state.process_context else ""
            )
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
    if page == "Upload Documents":
        render_upload_page()
    elif page == "Bulk Load (130+ Files)":
        render_bulk_load_page()
    elif page == "Define Context":
        render_context_page()
    elif page == "Analyze":
        render_analyze_page()
    elif page == "Review Process Document":
        render_process_document_page()
    elif page == "Gap Analysis":
        render_gap_analysis_page()
    elif page == "Chat":
        render_chat_page()

    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.caption("Document Analysis System v2.0 - With Session Management & Document Refinement")


if __name__ == "__main__":
    main()
