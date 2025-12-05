FROM python:3.10.13

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app
WORKDIR /app

RUN pip install -e .

CMD ["python", "perf_test/main.py"]