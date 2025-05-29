#!/bin/bash

# 数学变换可视化工具部署脚本

echo "开始部署数学变换可视化工具..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装Python依赖包..."
pip install --upgrade pip
pip install -r requirements.txt

# 进入Django项目目录
cd visual_website

# 收集静态文件
echo "收集静态文件..."
python manage.py collectstatic --noinput

# 运行数据库迁移
echo "运行数据库迁移..."
python manage.py makemigrations
python manage.py migrate

# 运行测试
echo "运行测试..."
python manage.py test

# 创建超级用户（可选）
echo "是否要创建超级用户？(y/n)"
read -r create_superuser
if [ "$create_superuser" = "y" ]; then
    python manage.py createsuperuser
fi

echo "部署完成！"
echo ""
echo "启动开发服务器："
echo "  cd visual_website"
echo "  python manage.py runserver"
echo ""
echo "生产环境部署："
echo "  使用 gunicorn 或 uwsgi 部署"
echo "  配置 nginx 反向代理"
echo "  使用 production_settings.py 配置文件"
