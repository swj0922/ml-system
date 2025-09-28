"""
启动FastAPI服务的脚本
"""

import os
import sys
import subprocess
import argparse

def install_dependencies():
    """安装项目依赖"""
    print("正在安装项目依赖...")
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    print("依赖安装完成")

def start_service(host="localhost", port=8000, reload=False):
    """启动FastAPI服务"""
    api_path = os.path.join(os.path.dirname(__file__), 'src', 'api_service.py')
    
    print(f"正在启动FastAPI服务，监听地址: {host}:{port}")
    print("API文档地址: http://{}:{}/docs".format(host, port))
    
    # 构建uvicorn命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api_service:app",
        "--host", host,
        "--port", str(port),
        "--app-dir", os.path.join(os.path.dirname(__file__), 'src')
    ]
    
    if reload:
        cmd.append("--reload")
    
    # 启动服务
    subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description="启动FastAPI机器学习模型预测服务")
    parser.add_argument("--install-deps", action="store_true", help="安装项目依赖")
    parser.add_argument("--host", default="localhost", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="监听端口 (默认: 8000)")
    parser.add_argument("--reload", action="store_true", help="启用热重载 (开发模式)")
    
    args = parser.parse_args()
    
    if args.install_deps:
        install_dependencies()
    
    start_service(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()