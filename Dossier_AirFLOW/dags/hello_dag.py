from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def hello_world():
    print("Hello World!")

with DAG(
    "test_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
) as dag:
    task = PythonOperator(
        task_id="hello_task",
        python_callable=hello_world,
    )
