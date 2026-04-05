@echo off
echo [1/3] Activating Virtual Environment...
call .\ai_research_env\Scripts\activate

echo [2/3] Checking GPU Status...
nvidia-smi

echo [3/3] Running Entropy Monitor...
python entropy_monitor.py

pause