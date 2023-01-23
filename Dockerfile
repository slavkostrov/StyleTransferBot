# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
COPY TransferBot TransferBot
COPY setup.py setup.py
COPY run.py run.py

RUN pip3 install -r requirements.txt
RUN pip3 install -e .

# ENV TG_STYLE_BOT_TOKEN
CMD ["python3", "run.py"]
