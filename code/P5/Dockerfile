FROM python:3.13-slim

WORKDIR /app

# 先安装依赖再复制业务代码，方便 Docker 构建缓存复用。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

ENV PYTHONUNBUFFERED=1
ENV AGENT_VERSION=mock-v1
ENV ENVIRONMENT=docker

# Agent 服务只暴露 FastAPI；Prometheus 和 Grafana 由 docker-compose 编排。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
