"""
Simple Streaming Agent Example

A basic calculator agent that demonstrates:
- Simple StreamingAgent implementation
- Progress updates via SSE
- Basic error handling
"""

from health_universe_a2a import MessageContext, StreamingAgent


class CalculatorAgent(StreamingAgent):
    """Simple calculator agent for demonstrating StreamingAgent basics."""

    def get_agent_name(self) -> str:
        return "Simple Calculator"

    def get_agent_description(self) -> str:
        return "Performs basic arithmetic calculations (add, subtract, multiply, divide)"

    async def process_message(self, message: str, context: MessageContext) -> str:
        """
        Process a calculation request.

        Expected format: "5 + 3" or "10 * 2", etc.
        """
        # Update progress - parsing
        await context.update_progress("Parsing expression...", 0.3)

        # Parse the expression
        try:
            parts = message.strip().split()
            if len(parts) != 3:
                return "Please use format: number operator number (e.g., '5 + 3')"

            num1 = float(parts[0])
            operator = parts[1]
            num2 = float(parts[2])

        except (ValueError, IndexError):
            return "Invalid format. Use: number operator number (e.g., '5 + 3')"

        # Update progress - calculating
        await context.update_progress("Calculating...", 0.7)

        # Perform calculation
        if operator == "+":
            result = num1 + num2
        elif operator == "-":
            result = num1 - num2
        elif operator == "*":
            result = num1 * num2
        elif operator == "/":
            if num2 == 0:
                return "Error: Cannot divide by zero"
            result = num1 / num2
        else:
            return f"Unknown operator: {operator}. Use +, -, *, or /"

        # Return final result
        return f"{num1} {operator} {num2} = {result}"


# Example usage (would integrate with your A2A server)
if __name__ == "__main__":
    print("Simple Calculator Agent Example")
    print("This agent would be integrated with your A2A server.")
    print("\nExample requests:")
    print("  '5 + 3' -> '5 + 3 = 8'")
    print("  '10 * 2' -> '10 * 2 = 20'")
    print("  '15 / 3' -> '15 / 3 = 5.0'")
