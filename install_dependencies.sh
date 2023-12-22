#!/bin/bash

# 确保脚本在遇到错误时终止
set -e

## 更新pip到最新版本
#pip install --upgrade pip

# 安装requirements.txt中的所有依赖
pip install -r requirements.txt

echo "Sucessfully install all dependencies!"
