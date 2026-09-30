# Production Dockerfile for EVlove EV Route Optimizer
FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose default Streamlit port
EXPOSE 8506

# Container healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8506/_stcore/health || exit 1

# Execute Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8506", "--server.address=0.0.0.0"]
