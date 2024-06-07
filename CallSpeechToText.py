import requests

# Replace with the URL where your FastAPI server is running
base_url = "http://192.168.18.88:2000"
# Replace with your API key
api_key = "apikey1"
def transcribe_video(file_path,language):
    try:
        with open(file_path, 'rb') as video_file:
            files = {'video_file': video_file}
            # video_file: UploadFile,news_type: str = None,language:str=None
            headers = {'api_key': api_key,'news_type':"live",'language':language}  # Include the API key in the header
            response = requests.post(f"{base_url}/transcribe_video/", files=files, headers=headers)

        if response.status_code == 200:
            result = response.json()
            # print(result)
            return result
           
    
        else:
            return ("Error:", response.text)
    except Exception as e:
        return ("Error:", str(e))

# if __name__ == "__main__":
#     video_path = r"G:\MWaqar\Speech-to-Text\uploads\b\chunks\chunk1.wav"  # Replace with your video file path
#     transcribe_video(video_path)
