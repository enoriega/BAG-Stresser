#!/usr/bin/env python3
"""
Simple script to enumerate available models from the server.

Uses API configuration from .env file and displays all available models.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from capacity_common import get_available_models


def main():
    """Main function to enumerate and display available models."""
    # Load environment variables from .env
    load_dotenv()

    # Get API configuration
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE')

    # Validate configuration
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)

    if not api_base:
        print("Error: OPENAI_API_BASE not found in .env file")
        sys.exit(1)

    print(f"API Base URL: {api_base}")
    print(f"API Key: {api_key[:20]}..." if len(api_key) > 20 else f"API Key: {api_key}")
    print()

    # Check for model filter file
    filter_file = "model_filter.txt"
    if Path(filter_file).exists():
        print(f"Model filter file found: {filter_file}")
        print("(Models in this file will be excluded from the list)")
    else:
        print(f"No model filter file found ({filter_file})")
    print()

    # Fetch available models
    print("Fetching available models from server...")
    available_models = get_available_models(api_key, api_base, filter_file=filter_file)

    if not available_models:
        print("Error: No models available or failed to fetch models")
        sys.exit(1)

    # Display results
    print(f"\n{'='*70}")
    print(f"AVAILABLE MODELS ({len(available_models)} total)")
    print(f"{'='*70}")

    for i, model_name in enumerate(sorted(available_models), 1):
        print(f"{i:3d}. {model_name}")

    print(f"{'='*70}")
    print(f"\nTotal: {len(available_models)} model(s) available")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        sys.exit(1)
