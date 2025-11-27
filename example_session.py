"""
Example of simulating a user session with repeated stress tests.
"""
import os
import asyncio
from dotenv import load_dotenv
from stresser import simulate_user_session, print_session_report

# Load environment variables
load_dotenv()

# Get API configuration
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
model_name = os.getenv('MODEL_NAME')  # If set, uses fixed model; if None, random selection

async def main():
    """Main async function to run the session simulation."""
    print("=" * 60)
    print("USER SESSION SIMULATION")
    print("=" * 60)
    print()

    # Run simulation for 30 seconds (can be adjusted)
    # Default is 60 seconds
    duration = 30
    concurrency = 3  # Run 3 conversations concurrently for better throughput

    if model_name:
        print(f"Using fixed model: {model_name}")
    else:
        print("Will randomly select models from API")

    print(f"Duration: {duration} seconds")
    print(f"Concurrency: {concurrency} concurrent conversations")
    print()

    # Run the simulation with concurrent execution
    stats = await simulate_user_session(
        conversations_dir='conversations',
        duration_seconds=duration,
        model_name=model_name,  # None = random selection
        api_key=api_key,
        api_base=api_base,
        temperature_range=(0.5, 1.0),  # Random temperature between 0.5 and 1.0
        concurrency=concurrency  # Run multiple conversations in parallel
    )

    # Print the detailed report
    print_session_report(stats)

    # Additional custom analysis
    print("\nADDITIONAL INSIGHTS")
    print("-" * 60)
    if stats.successful_conversations > 0:
        avg_messages_per_conversation = stats.total_messages_sent / stats.successful_conversations
        avg_tokens_per_conversation = stats.total_tokens / stats.successful_conversations
        print(f"Average messages per conversation: {avg_messages_per_conversation:.1f}")
        print(f"Average tokens per conversation: {avg_tokens_per_conversation:.0f}")
        print(f"Tokens per second: {stats.total_tokens / stats.duration_seconds:.1f}")

    success_rate = (stats.successful_conversations / stats.total_conversations * 100) if stats.total_conversations > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
