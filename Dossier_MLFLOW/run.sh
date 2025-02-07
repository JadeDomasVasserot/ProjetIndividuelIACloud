python -m venv mlflow-server
source mlflow-server/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mlflow server

docker build -t mlflow-server .
docker run \
-e PORT=5050 \
-p 5050:5050 \
-v "$(pwd):/home/app" \
-it  mlflow-server