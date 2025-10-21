@echo off
REM Create venv, activate, upgrade pip and install pinned packages (cmd)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install pandas==2.3.3 scikit-learn==1.7.2 joblib==1.3.2
python -c "import sys, pandas as pd, sklearn; from sklearn.ensemble import IsolationForest; print('PYTHON ->', sys.executable); print('pandas', pd.__version__, 'sklearn', sklearn.__version__)"