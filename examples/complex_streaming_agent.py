"""
Complex Streaming Agent Example

A data analysis agent that demonstrates:
- Validation with ValidationResult
- Multiple progress updates
- Artifact generation
- Lifecycle hooks
- Error handling
"""

import asyncio
import json
import statistics
from typing import Any

from health_universe_a2a import (
    AgentSkill,
    MessageContext,
    StreamingAgent,
    ValidationAccepted,
    ValidationRejected,
    ValidationResult,
)


class DataAnalysisAgent(StreamingAgent):
    """
    Complex data analysis agent demonstrating advanced features.

    Analyzes a list of numbers and generates:
    - Statistical summary
    - Data visualization artifacts
    - Detailed progress updates
    """

    def __init__(self) -> None:
        super().__init__()
        self.analysis_count = 0

    def get_agent_name(self) -> str:
        return "Data Analyzer Pro"

    def get_agent_description(self) -> str:
        return "Analyzes numerical datasets and generates statistical insights with visualizations"

    def get_agent_version(self) -> str:
        return "2.1.0"

    def get_provider_organization(self) -> str:
        """Custom provider organization."""
        return "Example Analytics Inc."

    def get_provider_url(self) -> str:
        """Custom provider URL."""
        return "https://example-analytics.com"

    def get_agent_skills(self) -> list[AgentSkill]:
        """
        Declare specific skills this agent provides.

        Skills allow fine-grained capability declaration beyond the general description.
        """
        return [
            AgentSkill(
                id="statistical_analysis",
                name="Statistical Analysis",
                description="Performs comprehensive statistical analysis on numerical datasets",
                tags=["statistics", "analytics", "data-science"],
                input_modes=["application/json"],
                output_modes=["application/json", "text/markdown"],
                examples=["[1, 2, 3, 4, 5]", "[10.5, 20.3, 30.1]"],
            ),
            AgentSkill(
                id="data_visualization",
                name="Data Visualization",
                description="Generates histogram and other visualization data",
                tags=["visualization", "charts", "histogram"],
                input_modes=["application/json"],
                output_modes=["application/json"],
                examples=["[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"],
            ),
            AgentSkill(
                id="report_generation",
                name="Report Generation",
                description="Creates human-readable analysis reports",
                tags=["reporting", "documentation", "markdown"],
                input_modes=["application/json"],
                output_modes=["text/markdown"],
                examples=["[100, 200, 300]"],
            ),
        ]

    def get_supported_input_formats(self) -> list[str]:
        """This agent only accepts JSON arrays."""
        return ["application/json"]

    def get_supported_output_formats(self) -> list[str]:
        """This agent outputs JSON and Markdown."""
        return ["application/json", "text/markdown"]

    async def validate_message(self, message: str, metadata: dict[str, Any]) -> ValidationResult:
        """Validate that message contains valid JSON array of numbers."""
        try:
            data = json.loads(message)

            if not isinstance(data, list):
                return ValidationRejected(reason="Input must be a JSON array of numbers")

            if len(data) == 0:
                return ValidationRejected(reason="Array must contain at least one number")

            if len(data) > 10000:
                return ValidationRejected(reason="Array too large (max 10,000 elements)")

            if not all(isinstance(x, (int, float)) for x in data):
                return ValidationRejected(reason="All elements must be numbers")

            # Estimate processing time (100 numbers per second)
            estimated_seconds = max(1, len(data) // 100)

            return ValidationAccepted(estimated_duration_seconds=estimated_seconds)

        except json.JSONDecodeError:
            return ValidationRejected(reason="Invalid JSON format")

    async def on_startup(self) -> None:
        """Initialize on startup."""
        self.logger.info("Data Analysis Agent starting up...")
        self.analysis_count = 0

    async def on_task_start(self, message: str, context: MessageContext) -> None:
        """Log task start."""
        self.analysis_count += 1
        self.logger.info(f"Starting analysis #{self.analysis_count} for user {context.user_id}")

    async def on_task_complete(self, message: str, result: str, context: MessageContext) -> None:
        """Log task completion."""
        self.logger.info(f"Analysis #{self.analysis_count} completed successfully")

    async def on_task_error(
        self, message: str, error: Exception, context: MessageContext
    ) -> str | None:
        """Handle errors gracefully."""
        self.logger.error(f"Analysis #{self.analysis_count} failed: {error}")

        if isinstance(error, asyncio.TimeoutError):
            return "Analysis timed out. Please try a smaller dataset."

        return None  # Use default error message

    async def process_message(self, message: str, context: MessageContext) -> str:
        """Analyze the dataset and generate insights."""

        # Parse data
        await context.update_progress("Parsing dataset...", 0.1)
        data = json.loads(message)

        # Simulate some processing time for demo
        await asyncio.sleep(0.5)

        # Calculate basic statistics
        await context.update_progress("Calculating statistics...", 0.3)

        stats = {
            "count": len(data),
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "min": min(data),
            "max": max(data),
        }

        # Add standard deviation if enough data points
        if len(data) > 1:
            stats["stdev"] = statistics.stdev(data)

        await asyncio.sleep(0.3)

        # Create summary artifact
        await context.update_progress("Generating summary...", 0.6)

        summary = {
            "dataset_size": len(data),
            "statistics": stats,
            "analysis_id": self.analysis_count,
        }

        await context.add_artifact(
            name="Statistical Summary",
            content=json.dumps(summary, indent=2),
            data_type="application/json",
            description="Comprehensive statistical analysis of the dataset",
        )

        await asyncio.sleep(0.2)

        # Create visualization data (simple histogram)
        await context.update_progress("Creating visualization...", 0.8)

        # Create 10 bins for histogram
        bins = 10
        bin_size = (stats["max"] - stats["min"]) / bins
        histogram = [0] * bins

        for value in data:
            bin_idx = min(int((value - stats["min"]) / bin_size), bins - 1)
            histogram[bin_idx] += 1

        viz_data = {
            "type": "histogram",
            "bins": bins,
            "bin_size": bin_size,
            "counts": histogram,
            "range": [stats["min"], stats["max"]],
        }

        await context.add_artifact(
            name="Histogram Data",
            content=json.dumps(viz_data, indent=2),
            data_type="application/json",
            description="Histogram visualization data",
        )

        await asyncio.sleep(0.2)

        # Create human-readable report
        await context.update_progress("Finalizing report...", 0.95)

        report = f"""# Data Analysis Report

## Dataset Overview
- **Size**: {stats["count"]} data points
- **Range**: {stats["min"]:.2f} to {stats["max"]:.2f}

## Statistical Summary
- **Mean**: {stats["mean"]:.2f}
- **Median**: {stats["median"]:.2f}
"""

        if "stdev" in stats:
            report += f"- **Standard Deviation**: {stats['stdev']:.2f}\n"

        report += f"""
## Key Insights
- The dataset shows a range of {stats["max"] - stats["min"]:.2f}
- The average value is {stats["mean"]:.2f}
- Half of the values are below {stats["median"]:.2f}

Analysis completed successfully! See attached artifacts for detailed statistics and visualization data.
"""

        await context.add_artifact(
            name="Analysis Report",
            content=report,
            data_type="text/markdown",
            description="Human-readable analysis report",
        )

        # Return final message
        return f"Analysis complete! Processed {len(data)} data points. Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}. See attached artifacts for full report."


# Example usage
if __name__ == "__main__":
    print("Complex Data Analysis Agent Example")
    print("\nFeatures demonstrated:")
    print("  - Message validation with custom error messages")
    print("  - Multiple progress updates throughout processing")
    print("  - Multiple artifacts (JSON, Markdown)")
    print("  - Lifecycle hooks (startup, task start/complete/error)")
    print("  - Error handling with custom messages")
    print("  - AgentCard creation with skills declaration")
    print("  - Custom provider information")
    print("  - Custom input/output format specification")
    print("\nExample request:")
    print("  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    print("\nWould generate:")
    print("  - Statistical Summary (JSON)")
    print("  - Histogram Data (JSON)")
    print("  - Analysis Report (Markdown)")

    # Demonstrate AgentCard creation
    print("\n" + "=" * 60)
    print("Agent Card Information:")
    print("=" * 60)

    agent = DataAnalysisAgent()
    card = agent.create_agent_card()

    print(f"\nName: {card.name}")
    print(f"Version: {card.version}")
    print(f"Description: {card.description}")
    print(f"Provider: {card.provider.organization} ({card.provider.url})")
    print(f"Protocol Version: {card.protocol_version}")
    print(f"Streaming: {card.capabilities.streaming}")
    print(f"Push Notifications: {card.capabilities.push_notifications}")

    print("\nInput Formats:")
    for fmt in card.default_input_modes:
        print(f"  - {fmt}")

    print("\nOutput Formats:")
    for fmt in card.default_output_modes:
        print(f"  - {fmt}")

    print(f"\nSkills ({len(card.skills)}):")
    for skill in card.skills:
        print(f"  - {skill.name}: {skill.description}")

    print("\n" + "=" * 60)
    print("The agent card can be served at /.well-known/agent-card.json")
    print("for agent discovery and integration.")
    print("=" * 60)
