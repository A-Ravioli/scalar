FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs (shared code)
COPY libs/ ./libs/

# Copy capacity_manager for SFCompute client
COPY services/capacity_manager/sfcompute_client.py ./capacity_manager/

# Copy service code
COPY services/forecasting/ ./

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "main.py"]

