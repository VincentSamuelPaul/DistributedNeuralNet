FROM python:3.12-slim
WORKDIR /app
RUN pip install flask flask-cors numpy prometheus-client --break-system-packages
COPY neuralnet/ ./neuralnet/
COPY inference/server.py .
COPY inference/frontend/ ./frontend/
ENV CHECKPOINT_PATH=/app/checkpoints/model.json
CMD ["python3", "server.py"]