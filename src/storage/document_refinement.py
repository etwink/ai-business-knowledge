"""Document refinement based on user feedback."""

from typing import List, Dict, Optional
from src.llm_integration import AzureLLMClient, PromptBuilder
from src.analyzers import ProcessDocument


class DocumentRefinement:
    """Refines documents based on user answers to clarification questions."""

    def __init__(self):
        self.llm = AzureLLMClient()

    def refine_process_document(
        self,
        process_document: ProcessDocument,
        questions: List[Dict[str, str]],
        user_answers: Dict[int, str]
    ) -> ProcessDocument:
        """
        Refine process document based on user answers.
        
        Args:
            process_document: Original process document
            questions: List of clarification questions
            user_answers: Mapping of question index to user's answer
            
        Returns:
            Refined ProcessDocument
        """
        # Build context from user answers
        answer_context = self._build_answer_context(questions, user_answers)
        
        # Refine each section
        refined_overview = self._refine_section(
            "Overview",
            process_document.overview,
            answer_context
        )
        
        refined_processes = self._refine_section(
            "Integrated Processes",
            process_document.integrated_processes,
            answer_context
        )
        
        refined_dependencies = self._refine_section(
            "Dependencies",
            process_document.dependencies,
            answer_context
        )
        
        refined_data_flow = self._refine_section(
            "Data Flow",
            process_document.data_flow,
            answer_context
        )
        
        refined_decision_points = self._refine_section(
            "Decision Points",
            process_document.decision_points,
            answer_context
        )
        
        refined_systems = self._refine_section(
            "Systems and Components",
            process_document.systems_and_components,
            answer_context
        )
        
        return ProcessDocument(
            overview=refined_overview,
            integrated_processes=refined_processes,
            dependencies=refined_dependencies,
            data_flow=refined_data_flow,
            decision_points=refined_decision_points,
            systems_and_components=refined_systems
        )

    def _build_answer_context(
        self,
        questions: List[Dict[str, str]],
        user_answers: Dict[int, str]
    ) -> str:
        """Build a formatted context from user answers."""
        context_parts = ["User-Provided Information:"]
        
        for idx, answer in user_answers.items():
            if idx < len(questions):
                question = questions[idx]['question']
                context_parts.append(f"\nQ: {question}")
                context_parts.append(f"A: {answer}")
        
        return "\n".join(context_parts)

    def _refine_section(
        self,
        section_name: str,
        section_content: str,
        answer_context: str
    ) -> str:
        """Refine a specific section with user feedback."""
        prompt = f"""You are a technical documentation expert. Enhance the following section with new information provided by the user.

Section: {section_name}

Original Content:
{section_content}

{answer_context}

Instructions:
1. Incorporate the user-provided information into the section
2. Maintain the original structure and style
3. Add clarifications and details from user answers
4. Ensure consistency with the existing content
5. Keep technical accuracy

Provide the refined section content:"""

        refined = self.llm.query(prompt, temperature=0.5)
        return refined

    def generate_refinement_summary(
        self,
        original_document: ProcessDocument,
        refined_document: ProcessDocument,
        user_answers: Dict[int, str]
    ) -> str:
        """Generate a summary of what was refined."""
        prompt = f"""Summarize the improvements made to the process document based on user feedback.

Original Document Summary:
{original_document.overview[:500]}

Refined Document Summary:
{refined_document.overview[:500]}

Key User Inputs:
{chr(10).join([f'- {answer}' for answer in list(user_answers.values())[:5]])}

Generate a concise summary (3-5 sentences) of the improvements made:"""

        return self.llm.query(prompt, temperature=0.5)
