FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy libs (shared code)
COPY libs/ ./libs/

# Copy service code
COPY services/billing/ ./

# Set Python path
ENV PYTHONPATH=/app

CMD ["python", "main.py"]

