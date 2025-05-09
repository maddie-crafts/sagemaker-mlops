FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY build_pipeline.py .
COPY utils/ utils/
COPY modules/ modules/
COPY src/ src/

CMD ["python", "build_pipeline.py"]