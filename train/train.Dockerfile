FROM python:3.8.12-slim

RUN apt update && apt install -y gcc g++ libboost-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY train-requirements.txt /
RUN python -m pip install -r train-requirements.txt

COPY hyperparam-tuning-requirements.txt /
RUN python -m pip install -r hyperparam-tuning-requirements.txt

COPY utils.py /
COPY train.py /

COPY config.py /
