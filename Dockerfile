FROM python:3.8-slim AS bot

ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PYTHONDONTWRITEBYTECODE 1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100

ENV TG_STYLE_BOT_TOKEN ${TG_STYLE_BOT_TOKEN}

RUN apt-get update
RUN apt-get install -y python3 python3-pip python-dev build-essential python3-venv

RUN mkdir -p /codebase /storage
ADD . /codebase
WORKDIR /codebase

COPY requirements.txt requirements.txt
COPY TransferBot /codebase/TransferBot
COPY setup.py /codebase/setup.py
COPY run.py /codebase/run.py

RUN pip3 install -r /requirements.txt
RUN pip3 install -e /codebase

CMD ["python3", "run.py"]
