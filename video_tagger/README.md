# Video Tagger

## Description
This project provides a REST API that processes video files by transcribing their content and tagging it using OpenAI's capabilities.

## How to Start Locally

### Prerequisites
Ensure you have `ffmpeg` installed. If not, install it with:
```bash
brew install ffmpeg
```

### Steps

#### 1. Set Up the Virtual Environment
Activate the virtual environment:
```bash
source ./venv/bin/activate
```

#### 2. Install the necessary packages
Install the required packages:
```bash
pip install -r requirements.txt
```

#### 3. Configure env vars
Create a `.env` file based on the `.env.example` template.

**Required Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key.
- `GOOGLE_STORAGE_BUCKET_NAME`: The name of your Google Cloud Storage bucket for uploading audio files (necessary for audio files longer than 5 minutes).
- `GOOGLE_APPLICATION_CREDENTIALS`: The path to your Google Cloud Account credentials file. 

To set up your Google Cloud credentials:
1. Follow [these instructions](https://cloud.google.com/iam/docs/service-accounts-create#creating) to create a service account.
2. Ensure the service account has access to Google Cloud Storage and the Speech-to-Text API.
3. Create a key for this service account and download the JSON credentials file.
4. Move the downloaded file to the root of the project and set the `GOOGLE_APPLICATION_CREDENTIALS` variable in your `.env` file to its path.

#### 4. Run the Project
To start the project in development mode:
```bash
fastapi dev main.py
```

To start the project in production mode:
```bash
uvicorn main:app
```

## Docker Image
You can also run the app using Docker:

1. Build the Docker image:
```bash
docker build -t video-tagger .
```

2. Run the Docker container:
```bash
docker run -p 8000:8000 video-tagger
```

Access the API documentation at `http://0.0.0.0:8000/docs`.

## Usage
The API provides a PUT endpoint at `/transcribe`. It accepts a multipart request with a file and an optional `service_provider` parameter (`google` or `openai`) to select the speech-to-text service.

To test the API, navigate to `http://localhost:8000/docs` in your browser, where you can interact with the API directly.