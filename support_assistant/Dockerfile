FROM python:3.11-slim

WORKDIR /app

COPY support_assistant/requirements.txt .

# Install CPU-only PyTorch to avoid downloading CUDA packages
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the remaining application dependencies
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    -r requirements.txt

COPY support_assistant ./support_assistant

EXPOSE 8000

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]