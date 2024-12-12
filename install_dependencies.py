import os
import sys
import subprocess
import ast
from importlib import util


def get_imports(file_path):
    imports = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        print(f"警告: 无法解析 {file_path}")

    return imports


def install_package(package):
    try:
        # 检查包是否已安装
        if util.find_spec(package) is None:
            print(f"正在安装 {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            print(f"成功安装 {package}")
    except:
        print(f"警告: {package} 安装失败")


def is_venv_directory(path):
    """检查是否是虚拟环境目录"""
    venv_indicators = ['venv', 'virtualenv', '.env', 'env']
    path_parts = path.lower().split(os.sep)
    return any(indicator in path_parts for indicator in venv_indicators)


def scan_project(project_path):
    # 标准库列表
    stdlib_list = sys.stdlib_module_names

    # 收集所有导入
    all_imports = set()

    # 扫描所有 .py 文件
    for root, dirs, files in os.walk(project_path):
        # 跳过虚拟环境目录
        if is_venv_directory(root):
            dirs[:] = []  # 清空dirs列表以阻止继续遍历子目录
            continue

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"扫描文件: {file_path}")
                imports = get_imports(file_path)
                all_imports.update(imports)

    # 过滤掉标准库
    third_party_imports = {imp for imp in all_imports
                           if imp not in stdlib_list}

    # 安装缺失的包
    for package in third_party_imports:
        install_package(package)


if __name__ == "__main__":
    project_path = "."
    print("开始扫描项目...")
    scan_project(project_path)
    print("完成！")
