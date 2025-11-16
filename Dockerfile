# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录为/app
WORKDIR /app

# 复制requirements.txt文件到/app路径下
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

# 复制应用程序代码
# (仅复制，如果文件夹中的代码更新不会随之更新，除非重新构建镜像。)
# 要想随时更新容器中的代码
# 就像docker-compose.yml中设置的那样，
# 需要在docker-compose.yml中设置volumes，将本地代码挂载到容器中
# 将宿主机上的 src/ 目录及其所有内容复制到Docker容器内的 ./src/ 目录中（即/app/src/目录）
COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/
COPY data/ ./data/

# 暴露端口，声明容器运行时监听8000端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONPATH=/app

# 容器启动时执行的默认命令
# 使用 uvicorn 启动FastAPI应用服务器，运行 src.api_service 模块中的 app 实例，绑定到所有网络接口( 0.0.0.0 )的8000端口
CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000"]