FROM tiangolo/uvicorn-gunicorn-starlette:python3.8

ENV PORT 5000
RUN echo "this is the port $PORT"
EXPOSE 5000

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir --requirement requirements.txt

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]