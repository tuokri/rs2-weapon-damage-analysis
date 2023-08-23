FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    git \
    libsqlite3-dev  \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app:server", "--chdir", \
     "rs2simulator", "--workers", "3", "--threads", "2", "--preload", \
     "--max-requests", "1000", "--max-requests-jitter", "250"]
