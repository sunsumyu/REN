@echo off
echo ============================================================
echo   医疗问答数据集生成系统 - 冗余根目录文件清理工具
echo ============================================================
echo.
echo 该脚本将清理已成功移入 tests/ 或 scripts/ 文件夹的根目录冗余文件。
echo 请确保您已经备份了需要保留的任何自定义修改。
echo.
set /p choice=是否确定清理冗余文件? (Y/N): 
if /i "%choice%" neq "Y" (
    echo 操作已取消。
    pause
    exit /b
)

echo.
echo 正在清理冗余文件...

del /q check_models_temp.py
del /q clean_dataset.py
del /q convert_and_clean_dataset.py
del /q eval_benchmark.json
del /q eval_models.py
del /q extract_last_planner.py
del /q inspect_huanyou.py
del /q inspect_removed.py
del /q inspect_think.py
del /q llm_purify_dataset.py
del /q run_evals.py
del /q scratch_check_first_line.py
del /q sort_existing_log.py
del /q stress_test_deepseek.py
del /q stress_test_gpt4o.py
del /q stress_test_qwen.py
del /q test_llm_post.py
del /q test_network.py
del /q test_prompt_versioning.py
del /q test_reasoning_extraction.py
del /q verify_purification.py

echo.
echo 🎉 冗余文件清理完毕！您的项目目录已整理得井井有条。
echo.
pause
