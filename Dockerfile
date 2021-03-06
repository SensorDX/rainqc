FROM python:2.7.12
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
ENV PYTHONPATH=.
CMD ["app/app.py"]
