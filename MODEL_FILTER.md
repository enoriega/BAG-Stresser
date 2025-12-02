# Model Filter Feature

## Overview

The model filter feature allows you to exclude specific models from being used during stress testing. This is useful for:

- Excluding expensive models that you don't want to test
- Filtering out deprecated or legacy models
- Removing embedding models or non-chat models
- Excluding models that are known to be slow or unstable

## Usage

### Creating a Filter File

1. Create a file named `model_filter.txt` in the project root (or use a custom filename)
2. Add one model name per line
3. Lines starting with `#` are treated as comments
4. Empty lines are ignored

**Example `model_filter.txt`:**
```
# Expensive models to exclude
gpt-4-32k
claude-opus-3

# Legacy models
gpt-3.5-turbo-0301

# Embedding models
text-embedding-ada-002
```

### Using with Single Session

The `example_session.py` and direct `stresser.py` usage will automatically use `model_filter.txt` if it exists:

```python
from stresser import simulate_user_session

stats = await simulate_user_session(
    conversations_dir='conversations',
    duration_seconds=60,
    model_name=None,  # Will randomly select from filtered models
    api_key=api_key,
    api_base=api_base,
    model_filter_file="model_filter.txt"  # Optional, defaults to model_filter.txt
)
```

### Using with Multi-Session

When running `multi_session.py`, use the `--model-filter` argument:

```bash
# Use default model_filter.txt
python multi_session.py --sessions 3 --duration 60

# Use a custom filter file
python multi_session.py --sessions 3 --duration 60 --model-filter my_custom_filter.txt

# With multiprocessing
python multi_session.py --sessions 3 --workers 4 --model-filter model_filter.txt
```

## How It Works

1. When `model_name` is `None` (random selection mode), the system fetches all available models from the API
2. The filter file is loaded and parsed
3. Models listed in the filter file are removed from the available models list
4. A random model is selected from the remaining filtered list

## Notes

- If the filter file doesn't exist, no filtering is applied (all models are available)
- If all models are filtered out, an error will be raised
- The filter is case-sensitive - model names must match exactly
- When using a specific `model_name` (not random), the filter is ignored
- A message will be printed showing how many models were filtered out

## Example

Starting with 20 available models and a filter file excluding 3 models:

```
Starting user session simulation for 60 seconds...
Filtered out 3 model(s) based on model_filter.txt
Available conversations: 10
Randomly selecting from 17 available models
Concurrency level: 3 concurrent conversation(s)
```

## Command Line Reference

### multi_session.py

```bash
python multi_session.py [options]

Options:
  --model-filter PATH    Path to file containing model names to exclude
                         (default: model_filter.txt)
```

### Full Example

```bash
# Create your filter file
cat > model_filter.txt << EOF
# Exclude expensive models
gpt-4-32k

# Exclude embedding models
text-embedding-ada-002
text-embedding-3-small
EOF

# Run stress test with filtering
python multi_session.py \
  --sessions 5 \
  --workers 2 \
  --duration 120 \
  --model-filter model_filter.txt
```
