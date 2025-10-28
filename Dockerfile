FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    gcc \
    git \
    libpq-dev \
    libsqlite3-dev  \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

# Remove psycopg to install optimized local build later.
RUN awk '!/psycopg/' requirements.txt > tmpfile && mv tmpfile requirements.txt

RUN pip install --upgrade pip --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

# Install psycopg3 optimized local build.
RUN pip install psycopg[c,pool] --no-cache-dir

COPY rs2simulator/ ./rs2simulator/
COPY gunicorn.conf.py .

EXPOSE 8000

CMD ["gunicorn", "app:server", "--chdir", \
     "rs2simulator", "--workers", "3", "--threads", "2", "--preload", \
     "--max-requests", "1000", "--max-requests-jitter", "250"]
