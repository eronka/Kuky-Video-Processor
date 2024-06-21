FROM python:3.12-slim

WORKDIR /app

ARG GCP_KEY

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the application code to the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN touch /app/key.json && echo $GCP_KEY | base64 --decode >> /app/key.json

EXPOSE 8000

# Command to run the application in development mode
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]