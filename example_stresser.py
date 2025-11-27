"""
Example usage of the conversation stress testing function.
"""
import os
import asyncio
from dotenv import load_dotenv
from stresser import run_conversation_stress_test

# Load environment variables from .env file
load_dotenv()

# Get API configuration from environment
api_key = os.getenv('OPENAI_API_KEY')
api_base = os.getenv('OPENAI_API_BASE')
model_name = os.getenv('MODEL_NAME', 'openai.gpt-oss-20b-1:0')
temperature = float(os.getenv('TEMPERATURE', '0.7'))

async def main():
    """Main async function to run the stress test."""
    # Run a single conversation test
    conversation_file = 'conversations/conversation_05.json'

    print(f"Running stress test on: {conversation_file}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: No limit (LLM can generate as much as it wants)")
    print("-" * 60)

    stats = await run_conversation_stress_test(
        conversation_file_path=conversation_file,
        model_name=model_name,
        temperature=temperature,
        max_tokens=None,  # No limit - let LLM generate as much as it wants
        max_messages=None,  # Send all messages
        api_key=api_key,
        api_base=api_base
    )

    # Print statistics
    print(f"\nConversation ID: {stats.conversation_id}")
    print(f"Total messages sent: {stats.total_messages_sent}")
    print(f"Total user messages: {stats.total_user_messages}")
    print(f"Total AI responses: {stats.total_ai_responses}")
    print(f"\nToken Statistics:")
    print(f"  Input tokens: {stats.total_tokens_input}")
    print(f"  Output tokens: {stats.total_tokens_output}")
    print(f"  Total tokens: {stats.total_tokens}")
    print(f"\nLatency Statistics:")
    print(f"  Total latency: {stats.total_latency_seconds:.3f}s")
    print(f"  Average latency: {stats.average_latency_seconds:.3f}s")
    print(f"  Min latency: {stats.min_latency_seconds:.3f}s")
    print(f"  Max latency: {stats.max_latency_seconds:.3f}s")
    print(f"  Total sleep time: {stats.total_sleep_time_seconds:.3f}s")
    print(f"\nPer-message latencies:")
    for i, latency in enumerate(stats.message_latencies, 1):
        print(f"  Message {i}: {latency:.3f}s")

    if stats.error:
        print(f"\nError: {stats.error}")
    else:
        print(f"\n✓ Conversation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
