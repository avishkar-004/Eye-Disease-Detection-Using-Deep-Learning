FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY Flask/ ./Flask/
EXPOSE 5000
CMD ["gunicorn", "--config", "Flask/gunicorn_config.py", "Flask.app:app"]
