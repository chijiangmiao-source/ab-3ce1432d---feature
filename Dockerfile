# 硅微条击中配对审计服务镜像（API 镜像从本 Dockerfile 生成）。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖以利用层缓存；运行时服务本身只依赖标准库。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY solver.py app.py verify.py ./
COPY tests ./tests

EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=5 \
    CMD python -c "import json,os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','8080'),timeout=3).read()" || exit 1

CMD ["python", "app.py"]
