# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY TransferBot TransferBot
COPY setup.py setup.py
COPY runner.py runner.py

RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["python3", "runner.py"]