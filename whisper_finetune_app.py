from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header

import moviepy.editor as mp
import os
import concurrent.futures
import asyncio
import re 
app = FastAPI()
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import split_audio
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v2"
model_id="Tensor_Model"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    # language='ur',
    # task="translate",
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    # max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    # return_timestamps=True,
    # no_timestamps_token_id=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Define a directory to store uploaded videos and audio
upload_dir = "uploads"

os.makedirs(upload_dir, exist_ok=True)

user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}

def process_video(video_file: UploadFile):
    try:
        # Read the uploaded video file into memory
        video_content = video_file.file.read()

        # Create a temporary file to save the uploaded video
        video_file_path = os.path.join(upload_dir, video_file.filename)
        with open(video_file_path, "wb") as temp_video_file:
            temp_video_file.write(video_content)

        # Use moviepy to process the saved video file
        video_clip = mp.VideoFileClip(video_file_path)

        # Extract audio from the video in memory
        audio_clip = video_clip.audio
        
        
        # Save the audio clip in the same directory as the video
        audio_file_path = os.path.join(upload_dir, f"{os.path.splitext(video_file.filename)[0]}.mp3")
        audio_clip.write_audiofile(audio_file_path)

        # Close the audio clip
        audio_clip.close()
        try:
            full_text_urdu=[]
            chunks_path=os.path.join(upload_dir,"chunks",os.path.splitext(video_file.filename)[0].strip())
            for audio_path in split_audio(input_file=audio_file_path,output_folder=chunks_path):
                result = pipe(audio_path)
                # print("own model")
                full_text_urdu.append(result["text"])

        except Exception as e:
            return {"error": f"Transcription error: {str(e)}"}

        try:
            # Remove the uploaded video and audio files
            video_clip.close()
            os.remove(audio_file_path)
            os.remove(video_file_path)
            import shutil
            shutil.rmtree(chunks_path)
        except Exception as e:
            return {"error": f"File delete error: {str(e)}"}

        # return {"urdu_full_text":full_text_urdu,"english_full_text":full_text_english}

        return {"urdu_full_text":"".join(full_text_urdu),"english_full_text":""}

    except Exception as e:
        return {"error": str(e)}
# Dependency to validate the API key
async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key



@app.post("/transcribe_video/")
async def transcribe_video_endpoint(
    video_file: UploadFile,
    api_key: str = Depends(get_api_key),  # Require API key for this route

):
    # Create a new thread for processing each user's video
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_video(video_file)
        )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1003,reload=True)
    
# run command in cmd 
# uvicorn whisper_finetune_app:app --host 0.0.0.0 --port 1003 --reload