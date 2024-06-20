from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from google.cloud import speech_v1p1beta1 as speech
from moviepy.editor import VideoFileClip
import tempfile
from dotenv import load_dotenv
from pydub import AudioSegment
from google.cloud import storage
import json
import time

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Setup Google Cloud credentials from environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


# Setup Google Cloud Storage client
storage_client = storage.Client()

class Tags(BaseModel):
    dislike: List[str]
    age: str
    gender: str
    interests: List[str]
    

class TranscriptionResponse(BaseModel):
    transcription: str
    tags: Tags

def transcribe_audio(audio_uri: str, service_provider: str = 'google'):
    if service_provider == 'google':
        print("Transcribing audio from URI using Google Cloud Speech-to-Text...")
        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(uri=audio_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result()

        transcripts = []
        for result in response.results:
            transcripts.append(result.alternatives[0].transcript)

        transcription = " ".join(transcripts)
        print("Audio transcription completed.")
        return transcription

    elif service_provider == 'openai':
        print("Transcribing audio from URI using OpenAI... (chunked)")
        return transcribe_audio_openai_chunked(audio_uri)

    else:
        raise HTTPException(status_code=400, detail="Invalid service provider. Supported values are 'google' or 'openai'.")

def transcribe_audio_openai_chunked(audio_path: str):
    print("Chunking audio file for OpenAI transcription...")
    try:
        audio = AudioSegment.from_file(audio_path)
        chunk_size = 25 * 1024 * 1024  # 25MB chunk size
        chunks = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            chunk_path = f"/tmp/audio_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav", codec='pcm_s16le', bitrate="16k", parameters=["-ar", "16000"])
            chunks.append(chunk_path)

        # Transcribe each chunk using OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        transcripts = []
        for chunk_path in chunks:
            with open(chunk_path, 'rb') as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
                transcripts.append(response.text)

        transcription = " ".join(transcripts)
        print("Audio transcription completed.")
        return transcription

    except Exception as e:
        print(f"Error during audio chunking and transcription: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during audio chunking and transcription.")

    finally:
        # Clean up temporary files
        for chunk_path in chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)



def extract_audio_from_video(video_file: UploadFile, service_provider: str) -> str:
    print("Starting audio extraction from video...")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.file.read())
            tmp_path = tmp.name
            print(f"Temporary video file written to: {tmp_path}")

        audio = AudioSegment.from_file(tmp_path)
        
        # Convert audio to mono if it's stereo
        if audio.channels > 1:
            print("Converting audio to mono...")
            audio = audio.set_channels(1)

        audio_path = tmp_path + ".wav"
        print(f"Writing mono audio to: {audio_path}")
        
        # Export audio as WAV using pydub
        audio.export(audio_path, format="wav", codec='pcm_s16le', bitrate="16k", parameters=["-ar", "16000"])

        if service_provider == 'google':
            # Upload audio file to Google Cloud Storage
            bucket_name = os.getenv('GOOGLE_STORAGE_BUCKET_NAME')
            storage_path = f"audio_files/{os.path.basename(audio_path)}"
            upload_audio_to_storage(audio_path, bucket_name, storage_path)

            # Get URI for the uploaded file
            audio_uri = f"gs://{bucket_name}/{storage_path}"
            print("Audio extraction and upload completed.")
            return audio_uri
        else:
            print("Audio extraction completed for OpenAI.")
            return audio_path

    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        raise HTTPException(status_code=400, detail="Could not process video file: " + str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"Temporary video file deleted: {tmp_path}")


def upload_audio_to_storage(file_path: str, bucket_name: str, storage_path: str):
    """Uploads a local file to Google Cloud Storage."""
    print(f"Uploading file {file_path} to GCS bucket {bucket_name} at {storage_path}")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(storage_path)
    blob.upload_from_filename(file_path)
    print(f"File uploaded to GCS")

def generate_tags(transcription: str):
    print(f"Generating tags for transcription {transcription}.")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": f"""
  Extract the following profiling information from the text: age, gender, interests.

  Text: "{transcription}"

  Response format:
  {{
    "dislike": ["dislike1", "dislike2"],
    "age": "extracted age",
    "gender": "extracted gender",
    "interests": ["interest1", "interest2"]
  }}"""
        }
      ],
      model="gpt-3.5-turbo",
    )
    # Extract the response content
    response_content = response.choices[0].message.content
    print(f"Response content: {response_content}")

    # Convert the response content to a Python dictionary
    profile_info = json.loads(response_content)
    print(f"Parsed profile info: {profile_info}")

    return profile_info


# @app.post("/transcribe/", response_model=TranscriptionResponse)
# async def transcribe(file: UploadFile = File(...)):
#     if file.content_type.startswith('video/'):
#         audio_uri = extract_audio_from_video(file)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
#     transcription = transcribe_audio(audio_uri)




#     tags = generate_tags(transcription)
#     return TranscriptionResponse(transcription=transcription, tags=tags)

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), service_provider: str = 'google'):
    if service_provider not in ['google', 'openai']:
        raise HTTPException(status_code=400, detail="Invalid service provider. Supported values are 'google' or 'openai'.")

    if file.content_type.startswith('video/'):
        audio_uri = extract_audio_from_video(file, service_provider)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    start_time = time.time()


    transcription = transcribe_audio(audio_uri, service_provider)
    tags = generate_tags(transcription)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"The transcribe function {elapsed_time:.4f} seconds to run.")

    return TranscriptionResponse(transcription=transcription, tags=tags)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
