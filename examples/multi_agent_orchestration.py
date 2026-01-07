"""
Multi-Agent Orchestration Example

This example demonstrates how to run multiple agents in a single server,
where agents can call each other using relative paths. This is useful for
orchestrator patterns where one agent coordinates the work of other agents.

Architecture:
- OrchestratorAgent: Coordinates document processing workflow
- DocumentReaderAgent: Reads and extracts text from documents
- SummarizerAgent: Summarizes extracted text
- ClassifierAgent: Classifies document type

All agents run on a single server (http://localhost:8501) at different paths:
- /orchestrator
- /document-reader
- /summarizer
- /classifier
"""

from health_universe_a2a import (
    Agent,
    AgentContext,
    serve_multi_agents,
)


class OrchestratorAgent(Agent):
    """
    Orchestrator that coordinates multiple agents to process documents.

    Workflow:
    1. Call document reader to extract text
    2. Call classifier to determine document type
    3. Call summarizer to create summary
    4. Return combined results
    """

    def __init__(self) -> None:
        super().__init__()
        # Reference other agents by their mount paths
        self.document_reader = "/document-reader"
        self.classifier = "/classifier"
        self.summarizer = "/summarizer"

    def get_agent_name(self) -> str:
        return "Document Orchestrator"

    def get_agent_description(self) -> str:
        return "Orchestrates document processing across multiple specialized agents"

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Process document through multi-agent pipeline."""

        # Step 1: Extract text from document
        await context.update_progress("Reading document...", 0.2)
        reader_response = await self.call_other_agent(
            self.document_reader, message, context, timeout=30.0
        )
        text = reader_response.text or ""

        # Step 2: Classify document type
        await context.update_progress("Classifying document...", 0.5)
        classifier_response = await self.call_other_agent(
            self.classifier, text, context, timeout=10.0
        )
        doc_type = classifier_response.text or "Unknown"

        # Step 3: Generate summary
        await context.update_progress("Generating summary...", 0.8)
        summary_response = await self.call_other_agent(self.summarizer, text, context, timeout=30.0)
        summary = summary_response.text or ""

        # Return combined results
        await context.update_progress("Complete!", 1.0)

        result = f"""Document Processing Complete

Document Type: {doc_type}

Summary:
{summary}

Original Text Length: {len(text)} characters
"""
        return result


class DocumentReaderAgent(Agent):
    """Reads documents and extracts text."""

    def get_agent_name(self) -> str:
        return "Document Reader"

    def get_agent_description(self) -> str:
        return "Extracts text content from documents"

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Extract text from document path."""
        await context.update_progress("Opening document...", 0.3)

        # Simulate document reading
        # In real implementation, would use PyPDF2, docx, etc.
        extracted_text = f"[Extracted text from: {message}]\n\nThis is the document content..."

        await context.update_progress("Extraction complete", 1.0)
        return extracted_text


class ClassifierAgent(Agent):
    """Classifies document type based on content."""

    def get_agent_name(self) -> str:
        return "Document Classifier"

    def get_agent_description(self) -> str:
        return "Classifies documents by type (report, invoice, letter, etc.)"

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Classify document based on text content."""
        await context.update_progress("Analyzing content...", 0.5)

        # Simple keyword-based classification (in real app, would use ML model)
        text_lower = message.lower()

        if "invoice" in text_lower or "bill" in text_lower:
            doc_type = "Invoice"
        elif "report" in text_lower or "summary" in text_lower:
            doc_type = "Report"
        elif "dear" in text_lower or "sincerely" in text_lower:
            doc_type = "Letter"
        else:
            doc_type = "General Document"

        await context.update_progress("Classification complete", 1.0)
        return doc_type


class SummarizerAgent(Agent):
    """Generates summaries of document text."""

    def get_agent_name(self) -> str:
        return "Document Summarizer"

    def get_agent_description(self) -> str:
        return "Generates concise summaries of document content"

    async def process_message(self, message: str, context: AgentContext) -> str:
        """Generate summary of document text."""
        await context.update_progress("Generating summary...", 0.6)

        # Simple extractive summary (first sentence + word count)
        # In real app, would use LLM or extractive summarization
        first_sentence = message.split(".")[0] if "." in message else message[:100]
        word_count = len(message.split())

        summary = f"{first_sentence}...\n\n[{word_count} words total]"

        await context.update_progress("Summary complete", 1.0)
        return summary


def main() -> None:
    """Start multi-agent server with all agents mounted."""

    # Create agent instances
    orchestrator = OrchestratorAgent()
    document_reader = DocumentReaderAgent()
    classifier = ClassifierAgent()
    summarizer = SummarizerAgent()

    # Serve all agents in a single server
    print("\n" + "=" * 60)
    print("Multi-Agent Orchestration Example")
    print("=" * 60)
    print("\nStarting server with 4 agents...")
    print("\nTest with:")
    print("  curl -X POST http://localhost:8501/orchestrator/ \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        '    -d \'{"jsonrpc": "2.0", "method": "message/send", "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "/path/to/document.pdf"}]}}, "id": 1}\''
    )
    print("\n" + "=" * 60 + "\n")

    serve_multi_agents(
        agents={
            "/orchestrator": orchestrator,
            "/document-reader": document_reader,
            "/classifier": classifier,
            "/summarizer": summarizer,
        },
        port=8501,
        log_level="info",
    )


if __name__ == "__main__":
    main()
