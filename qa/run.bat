@chcp 65001 >nul
@echo off

title 医疗多视角（Facets）多轮问答数据集生成器

echo ============================================================

echo      医疗多视角（Facets）多轮问答数据集生成系统

echo ============================================================

echo.

echo [1/2] 检查 Python 运行依赖包...

pip install -r requirements.txt

if %errorlevel% neq 0 (

    echo [ERROR] 依赖包装载失败，请检查 Python 安装与网络！

    pause

    exit /b %errorlevel%

)

echo [OK] 依赖检查完毕。

echo.

echo [2/2] 启动多轮对话问答数据生成流程...

python main.py

echo.

echo 流程结束。

pause

