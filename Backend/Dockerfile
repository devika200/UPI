# Python 3.14 – use requirements.txt versions that support it (pandas>=2.3.3, numpy 2.x, etc.)
FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Render expects the app to listen on PORT
ENV PORT=10000
EXPOSE 10000

CMD gunicorn --bind 0.0.0.0:${PORT} mainapp:app
