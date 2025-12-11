FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

COPY src/crawler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install

COPY src/crawler/ .
# utils are mounted via docker-compose, but for build context we might need to copy if we weren't mounting. 
# Since docker-compose mounts it, we are good for dev. 
# But for production build, we should copy. 
# The docker-compose context is root, so we can copy utils.
COPY utils/ ./utils/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
