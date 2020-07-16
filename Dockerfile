FROM python:3.6.7

COPY . /root
WORKDIR /root
# COPY requirements.txt /root
RUN python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
RUN pip install --no-cache-dir -r requirements.txt

