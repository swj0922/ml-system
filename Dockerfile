# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用程序代码
# (仅复制，如果文件夹中的代码更新不会随之更新，除非重新构建镜像。)
# 要想随时更新容器中的代码
# 就像docker-compose.yml中设置的那样，
# 需要在docker-compose.yml中设置volumes，将本地代码挂载到容器中
COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/
COPY data/ ./data/

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONPATH=/app

# 启动命令
CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000"]