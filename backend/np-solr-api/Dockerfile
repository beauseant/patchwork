
FROM python:3.9

WORKDIR /np-solr-api

COPY . .

RUN mkdir -p /data/source
RUN mkdir -p /data/inference

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dask[dataframe]"

EXPOSE 82

CMD ["python3", "app.py"]
