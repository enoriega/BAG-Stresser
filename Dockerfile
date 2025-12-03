# Use the official Python UV image (slim variant for smaller size)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync --no-dev

# Copy the rest of the application (including conversations directory and model_filter.txt if present)
COPY . .

# Create a default empty model_filter.txt if it doesn't exist
RUN touch /app/model_filter.txt

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the shell script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command arguments (can be overridden)
CMD ["--duration", "60", "--sessions", "1"]
