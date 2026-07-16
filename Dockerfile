FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

COPY *.py .

RUN mkdir -p /app/output

CMD ["python", "demo.py", "--plot", "--output-dir", "output"]
