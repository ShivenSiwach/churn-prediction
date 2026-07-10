# Base image
FROM python:3.11-slim

# Create a non-root user for security compliance
RUN adduser --disabled-password --gecos '' mluser

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies (clearing cache to keep image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Transfer ownership to the non-root user
RUN chown -R mluser:mluser /app

# Switch to the secure non-root user
USER mluser

# Expose port
EXPOSE 8000

# Run the API (Adjust api.main:app to main:app if your main.py is in the root directory)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
