@chcp 65001 >nul
@echo off
echo ============================================================
echo   Medical QA Dataset Generator - Workspace Cleanup Tool
echo ============================================================
echo.
echo This tool will safely remove redundant root files that have
echo been successfully moved into the tests/ and scripts/ folders.
echo.
set /p choice="Do you want to delete the redundant files? (Y/N): "
if /i "%choice%" neq "Y" (
    echo Cancelled.
    pause
    exit /b
)

echo.
echo Deleting redundant root files...

del /q check_models_temp.py
del /q clean_dataset.py
del /q convert_and_clean_dataset.py
del /q eval_benchmark.json
del /q eval_models.py
del /q extract_last_planner.py
del /q guideline_db.py
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
echo Cleanup complete! Your workspace root is now clean and tidy.
echo.
pause
