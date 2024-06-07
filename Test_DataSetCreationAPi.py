import requests
import os
import zipfile

# Replace with the URL where your FastAPI server is running
base_url = "http://192.168.18.119:1015"
# Replace with your API key
api_key = "apikey1"

def transcribe_video(file_path):
    try:
        file_name = os.path.basename(file_path)
        file_name_no_ext, _ = os.path.splitext(file_name)  # Extract file name without extension

        with open(file_path, 'rb') as video_file:
            files = {'video_file': video_file}
            headers = {'api_key': api_key, 'language': 'urdu'}  # Include the API key and language in the header
            response = requests.post(f"{base_url}/STTDataset/", files=files, headers=headers)

        if response.status_code == 200:
            # Save the zip file received in the response
            zip_file_path = f"{file_name}.zip"
            with open(zip_file_path, "wb") as zip_file:
                zip_file.write(response.content)
            
            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(file_name_no_ext)  # Extract to a folder with the same name as the input file

            print("Zip file contents extracted successfully.")
        else:
            print("Error:", response.text)
    except Exception as e:
        print("Error:", str(e))
    finally:
        # Delete the zip file after extraction
        os.remove(zip_file_path)

if __name__ == "__main__":
    video_path = r"D:\a.wav"  # Replace with your video file path
    transcribe_video(video_path)
