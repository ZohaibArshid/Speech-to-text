from fastapi import FastAPI, UploadFile, Depends, HTTPException, Header
import requests
import socket
import moviepy.editor as mp
import os
import concurrent.futures
import asyncio
from utils import *
import shutil
from CallSpeechToText import transcribe_video as TranscriptionCall
from silence import get_large_audio_chunks_on_silence 
from fastapi.responses import FileResponse,JSONResponse
app = FastAPI()

# Define a directory to store uploaded videos and audio
upload_dir = "uploads"

os.makedirs(upload_dir, exist_ok=True)

user_api_keys = {
    "user1": "apikey1",
    "user2": "apikey2",
    # Add more users and their API keys as needed
}


    # print(f"Text file created: {text_file_path}")
def process_video(file: UploadFile,language:str=None):
    try:
        # Read the uploaded video file into memory
        file_content = file.file.read()
        file_name, file_extension = os.path.splitext(file.filename)
      # Create a folder for the file inside the "uploads" directory
        file_folder = os.path.join(upload_dir, file_name)
        os.makedirs(file_folder, exist_ok=True)

        # Save the uploaded file in the file folder
        file_path = os.path.join(file_folder, file.filename)
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)

         # Check if the file has a video extension
        video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv"]
        if any(file_extension.lower().endswith(ext) for ext in video_extensions):
        # Save the audio clip in the same directory as the video
            audio_file_path = os.path.join(file_folder, f"{file_name}.wav")
            if not os.path.exists(audio_file_path):
                # Use moviepy to process the saved video file
                video_clip = mp.VideoFileClip(file_path)

                # Extract audio from the video in memory
                audio_clip = video_clip.audio

                audio_clip.write_audiofile(audio_file_path)
                video_clip.close()
                # Close the audio clip
                audio_clip.close()
            else:
                pass
        else:
            audio_file_path=file_path
        
        chunks=get_large_audio_chunks_on_silence(audio_file_path,file_folder)
        for audio in chunks:
            text=TranscriptionCall(audio,language=language)
            if language.lower()=='urdu':
                text=text['urdu_full_text']
            else:
                text=text['english_full_text']
            # print(text)
            create_text_file(audio,text)
          
        try:
            if file_extension in video_extensions:
                # print("file found")
                # print(file_path)
                # Remove the uploaded video and audio files
                video_clip.close()
                os.remove(file_path)
                # print(audio_file_path)
                os.remove(audio_file_path)
            else:
                os.remove(audio_file_path)
        except Exception as e:
            return {"error": "File delete error"}
        zip_folder=create_zip_folder(file_folder)
        response=FileResponse(zip_folder, filename=f'{file_name}.zip')
        # os.remove(f"{video_name}.zip")
        print(file_folder)
        shutil.rmtree(file_folder, ignore_errors=True)
        return response

    

    except Exception as e:
        return {"error": f"Internal Error{str(e)}"}
# Dependency to validate the API key
async def get_api_key(api_key: str = Header(None, convert_underscores=False)):
    if api_key not in user_api_keys.values():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# async def get_language_key(language: str = Header(None, convert_underscores=False)):
#     if language.lower() not in language_list.keys():
#         raise HTTPException(status_code=401, detail="Invalid language")
#     return language_list[language.lower()]
async def get_language_key(language: str = Header(None, convert_underscores=False)):
    return language
news_type_list=["live","dataset","report"]
async def get_news_type(news_type: str = Header(None, convert_underscores=False)):
    if news_type.lower() not in news_type_list:
        raise HTTPException(status_code=401, detail="Invalid news_type value")
    return news_type.lower()

@app.post("/STTDataset/")
async def STTDataset_endpoint(
    video_file: UploadFile,
    api_key: str = Depends(get_api_key),  # Require API key for this route
    language : str = Depends(get_language_key)
):
    # Create a new thread for processing each user's video
    # print(language)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: process_video(video_file,language)
        )
    return result



# host_name = socket.gethostname()
# URDU_API_BASE_URL=socket.gethostbyname(host_name)
# # Replace with the URL where your FastAPI server is running
# base_url = "http://"+URDU_API_BASE_URL+":2000"

# def check_api_status(base_url):
#     try:
#         # Use a short timeout, for example, 5 seconds
#         response = requests.get(base_url + "/docs", timeout=5)
        
#         # Check if the status code is in the range 200-299, indicating a successful request
#         if response.ok:
#             return True
#         else:
#             return False

#     except requests.RequestException as e:
#         return (f"Could not connect to the API at {base_url}")

# @app.get("/handshake/")
# async def handshake_endpoint(

# ):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         result = await asyncio.get_event_loop().run_in_executor(
#             executor,
#             lambda: check_api_status(base_url)
#         )
#     return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1015,reload=True)
    
# run command in cmd 
# uvicorn DatasetCreationAPi:app --host 0.0.0.0 --port 1015 --reload