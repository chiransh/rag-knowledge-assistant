# Base image — use GPU variant if available, CPU otherwise
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Set working directory
WORKDIR /app

# Copy dependency manifest first to leverage Docker layer caching
COPY requirements.txt .

# Install system dependencies (Java required for Apache Tika text extraction)
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get -y autoremove

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY rag.py .

# Expose Streamlit default port
EXPOSE 8501

# Start the Streamlit application
ENTRYPOINT ["streamlit", "run", "rag.py", "--server.port=8501", "--server.address=0.0.0.0"]
