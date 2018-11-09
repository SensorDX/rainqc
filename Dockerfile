FROM python:2.7
COPY . /rainqc
WORKDIR /rainqc
RUN pip install -r requirements.txt
EXPOSE 8000
CMD python ./rqc_main.py