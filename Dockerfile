FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py theme.py data.py trends.py maps.py ontology.py chat.py ./
COPY assets/ assets/
COPY data/ data/
COPY .streamlit/config.toml .streamlit/config.toml

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
