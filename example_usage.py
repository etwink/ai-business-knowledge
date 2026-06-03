"""Example script demonstrating programmatic usage of the Document Analysis System."""

from pathlib import Path
from src.document_loaders import get_loader
from src.analyzers import (
    DocumentAnalyzer,
    ProcessDocumentBuilder,
    GapAnalyzer,
    ClarificationQuestionGenerator
)


def example_analysis(document_paths):
    """
    Example function showing how to use the system programmatically.

    Args:
        document_paths: List of Path objects or strings to documents
    """
    print("=" * 80)
    print("Document Analysis System - Programmatic Example")
    print("=" * 80)

    # Step 1: Load and analyze individual documents
    print("\n📚 STEP 1: Analyzing individual documents...")
    print("-" * 80)

    analyzer = DocumentAnalyzer()
    analyses = []

    for doc_path in document_paths:
        doc_path = Path(doc_path)
        print(f"\nAnalyzing: {doc_path.name}")

        try:
            loader = get_loader(doc_path)
            document = loader.load()
            analysis = analyzer.analyze_document(document)
            analyses.append(analysis)

            print(f"  ✓ Processed successfully")
            print(f"  Summary: {analysis.summary[:200]}...")
            print(f"  Key Processes: {', '.join(analysis.key_processes[:3])}")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

    if not analyses:
        print("No documents were successfully analyzed.")
        return

    # Step 2: Generate comprehensive process document
    print("\n\n📖 STEP 2: Generating comprehensive process document...")
    print("-" * 80)

    builder = ProcessDocumentBuilder()
    process_doc = builder.build_process_document(analyses)

    print("\nProcess Document Generated:")
    print(f"\nOverview:\n{process_doc.overview[:300]}...")
    print(f"\nKey Dependencies:\n{process_doc.dependencies[:300]}...")

    # Step 3: Analyze gaps
    print("\n\n⚠️  STEP 3: Analyzing gaps and missing information...")
    print("-" * 80)

    gap_analyzer = GapAnalyzer()
    gaps = gap_analyzer.analyze_gaps(process_doc)

    total_gaps = sum(len(v) for v in gaps.gaps_by_category.values())
    print(f"\nGaps Identified: {total_gaps} across {len(gaps.gaps_by_category)} categories")
    for cat, items in list(gaps.gaps_by_category.items())[:3]:
        print(f"\n  • {cat}: {len(items)} identified")
        for item in items[:3]:
            print(f"    - {item}")

    if gaps.edge_cases:
        print(f"\n  • Edge Cases: {len(gaps.edge_cases)} identified")
        for item in gaps.edge_cases[:3]:
            print(f"    - {item}")

    # Step 4: Generate clarification questions
    print("\n\n❓ STEP 4: Generating clarification questions...")
    print("-" * 80)

    question_gen = ClarificationQuestionGenerator()
    questions = question_gen.generate_questions(process_doc, gaps)

    print(f"\nGenerated {len(questions)} clarification questions:")
    for idx, q in enumerate(questions[:5], 1):
        print(f"\n  {idx}. {q['question']}")
        if q['rationale']:
            print(f"     Why: {q['rationale'][:100]}...")

    # Step 5: Export results
    print("\n\n💾 STEP 5: Exporting results...")
    print("-" * 80)

    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    # Export process document
    with open(output_dir / "process_document.md", "w") as f:
        f.write("# Process Document\n\n")
        f.write(f"## Overview\n{process_doc.overview}\n\n")
        f.write(f"## Integrated Processes\n{process_doc.integrated_processes}\n\n")
        f.write(f"## Dependencies\n{process_doc.dependencies}\n\n")
        f.write(f"## Data Flow\n{process_doc.data_flow}\n\n")
        f.write(f"## Decision Points\n{process_doc.decision_points}\n\n")
        f.write(f"## Systems & Components\n{process_doc.systems_and_components}\n")

    # Export gap analysis
    with open(output_dir / "gap_analysis.md", "w") as f:
        f.write("# Gap Analysis\n\n")
        for cat, items in gaps.gaps_by_category.items():
            f.write(f"## {cat}\n")
            for item in items:
                f.write(f"- {item}\n")
            f.write("\n")
        if gaps.edge_cases:
            f.write("## Edge Cases\n")
            for item in gaps.edge_cases:
                f.write(f"- {item}\n")
            f.write("\n")
        if gaps.resource_gaps:
            f.write("## Resource Gaps\n")
            for item in gaps.resource_gaps:
                f.write(f"- {item}\n")

    # Export questions
    with open(output_dir / "clarification_questions.md", "w") as f:
        f.write("# Clarification Questions\n\n")
        for idx, q in enumerate(questions, 1):
            f.write(f"## Question {idx}\n{q['question']}\n\n")
            if q['rationale']:
                f.write(f"**Why:** {q['rationale']}\n\n")

    print(f"✓ Results exported to: {output_dir.absolute()}")
    print("  - process_document.md")
    print("  - gap_analysis.md")
    print("  - clarification_questions.md")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Example: Analyze files in sample_documents directory
    sample_dir = Path("sample_documents")

    if sample_dir.exists():
        documents = list(sample_dir.glob("**/*"))
        documents = [d for d in documents if d.is_file()]

        if documents:
            example_analysis(documents)
        else:
            print("No documents found in sample_documents directory")
    else:
        print("sample_documents directory not found")
        print("Create sample_documents folder and add documents to analyze")
