import time
import requests # type: ignore
from api_secrets import API_KEY_ASSEMBLYAI # type: ignore
import sys
upload_endpoint = "https://api.assemblyai.com/v2/upload"
headers = {'authorization': API_KEY_ASSEMBLYAI}
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
# Upload function
# uploads an audio file to AssemblyAI and returns the audio URL
def upload(filename):

    def read_file(filename, chunk_size=5242880):
        with open(filename, "rb") as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    upload_response = requests.post(
        upload_endpoint, headers=headers, data=read_file(filename)
    )

    audio_url = upload_response.json()["upload_url"]
    return audio_url
# Transcribe function
# sends an audio URL to AssemblyAI and returns the job ID
def transcribe(audio_url):

    transcript_request = {"audio_url": audio_url}  # type: ignore
    trancript_response = requests.post(
        transcript_endpoint, json=transcript_request, headers=headers
    )
    job_id = trancript_response.json()["id"]
    return job_id


filename = "C:/Users/HP/Documents/PFE/project/Output.wav"
# Poll function
# checks the status of a transcription job
def poll(job_id):
    polling_endpoint = transcript_endpoint + "/" + job_id
    polling_response = requests.get(polling_endpoint, headers=headers)
    return polling_response.json()
# Get transcription result URL function
# polls AssemblyAI for the transcription result and returns the result and error (if any)
def get_transcription_result_url(audio_url):
    job_id = transcribe(audio_url)
    while True:
        data = poll(job_id)
        if data["status"] == "completed":
            return data, None
        elif data["status"] == "error":
            return data, data["error"]
        
        print("Waiting 17 sec ......")
        time.sleep(17)
# Save transcript function
def save_transcript(audio_url):
    data, error = get_transcription_result_url(audio_url)
    
    print("Data:", data)
    print("Error:", error)
    
    if data:
        text_filename = filename + '.txt'
        with open(text_filename, 'w') as f:
            f.write(data['text'])
        print("Transcription saved!!")
    elif error:
        print('Error')
        
audio_url = upload(filename)
save_transcript(audio_url)