@echo off
python src\generate_demo_data.py
python src\train_xgboost.py
streamlit run app\app.py
