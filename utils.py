# from pydub import AudioSegment
# import os
# from datetime import timedelta

# def calculate_total_duration(root_folder):
#     total_duration = 0

#     # Iterate through each subfolder in the root folder
#     for subdir, dirs, files in os.walk(root_folder):
#         for file in files:
#             # Check if the file is an audio file (you can add more file extensions if needed)
#             if file.lower().endswith(('.mp3', '.wav', '.ogg', '.flac')):
#                 # Get the full path of the audio file
#                 file_path = os.path.join(subdir, file)

#                 # Load the audio file using pydub
#                 audio = AudioSegment.from_file(file_path)

#                 # Get the duration in seconds
#                 duration_seconds = audio.duration_seconds

#                 # Add the duration of the audio file to the total duration
#                 total_duration += duration_seconds
#                 # print("audio path:",file_path)
#                 # print("duration:",duration_seconds)
#     return total_duration

# # Specify the root folder of your dataset
# root_folder = "Dataset\Custom_dataset\common_voice"

# # # Call the function to calculate the total duration
# total_duration_seconds = calculate_total_duration(root_folder)
# # Calculate hours, minutes, and remaining seconds
# total_hours = int(total_duration_seconds // 3600)
# total_minutes = int((total_duration_seconds % 3600) // 60)
# total_seconds = int(total_duration_seconds % 60)
# # Format the output
# output = f"{total_hours} hours, {total_minutes} minutes, and {total_seconds} seconds"
# print(f"Total duration: {output}")

import os

def remove_empty_text_files(base_directory):
    audio_extensions = [".wav", ".mp3", ".flac"]  # Add more extensions if needed
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            for ext in audio_extensions:
                if file.endswith(ext):
                    audio_file_path = os.path.join(root, file)
                    text_file_path = os.path.join(root, file.replace(ext, ".txt"))

                    if os.path.exists(text_file_path) and os.path.getsize(text_file_path) == 0:
                        print(f"Removing empty files: {audio_file_path} and {text_file_path}")
                        os.remove(audio_file_path)
                        os.remove(text_file_path)
                    
                    # Check if the text file exists but audio file does not
                    if not os.path.exists(audio_file_path) and os.path.exists(text_file_path):
                        print(f"Removing orphan text file: {text_file_path}")
                        os.remove(text_file_path)

                    # Check if the audio file exists but text file does not
                    if os.path.exists(audio_file_path) and not os.path.exists(text_file_path):
                        print(f"Removing orphan audio file: {audio_file_path}")
                        os.remove(audio_file_path)
    
    # Remove empty directories
    for root, dirs, files in os.walk(base_directory, topdown=False):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            if not os.listdir(folder_path):
                print(f"Removing empty folder: {folder_path}")
                os.rmdir(folder_path)

# Replace 'your_dataset_directory' with the actual path to your dataset directory
remove_empty_text_files(base_directory=r"Dataset/Custom_dataset")

# import os
# from pydub import AudioSegment

# def split_audio(input_file, output_folder="chunks", chunk_duration=15):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Load audio file
#     audio = AudioSegment.from_file(input_file)

#     # Get the duration of the audio in seconds
#     audio_duration = len(audio) / 1000  # Convert milliseconds to seconds

#     # Calculate the number of full chunks
#     num_full_chunks = int(audio_duration / chunk_duration)
#     chunks_path = []

#     # Split audio into full chunks
#     for i in range(num_full_chunks):
#         start_time = i * chunk_duration * 1000  # Convert seconds to milliseconds
#         end_time = (i + 1) * chunk_duration * 1000

#         # Extract the chunk
#         chunk = audio[start_time:end_time]

#         # Save the chunk to the output folder
#         output_file = os.path.join(output_folder, f"chunk_{i + 1}.wav")
#         chunks_path.append(output_file)
#         chunk.export(output_file, format="wav")

#     # Handle the last chunk
#     start_time = num_full_chunks * chunk_duration * 1000
#     end_time = audio_duration * 1000
#     last_chunk = audio[start_time:end_time]

#     # Save the last chunk to the output folder
#     output_file = os.path.join(output_folder, f"chunk_{num_full_chunks + 1}.wav")
#     chunks_path.append(output_file)
#     last_chunk.export(output_file, format="wav")

#     # print(f"{num_full_chunks + 1} chunks created in {output_folder}")
#     return chunks_path
# import os
# import soundfile as sf  # You can use the `soundfile` library to get sample rate

# def print_sample_rates(root_dir):
#     for subdir, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.wav'):
#                 audio_path = os.path.join(subdir, file)
#                 try:
#                     # Get the sample rate using the soundfile library
#                     sample_rate = sf.info(audio_path).samplerate
#                     print(f"Audio Path: {audio_path}, Sample Rate: {sample_rate}")
#                 except Exception as e:
#                     print(f"Error processing {audio_path}: {e}")

# Example usage

# dataset_directory = "P Dataset"
# print_sample_rates(dataset_directory)
def create_text_file(audio_file_path,text):
    # Extract filename without extension
    filename_no_ext = os.path.splitext(audio_file_path)[0]

    # Create text file path
    text_file_path = filename_no_ext + '.txt'

    # Write audio file path to text file
    with open(text_file_path, 'w',encoding='utf-8') as text_file:
        text_file.write(text)
        
import os
import zipfile
def create_zip_folder(folder_path):
    # Get the parent directory of the folder
    parent_dir = os.path.dirname(folder_path)

    # Define the name for the zip file
    zip_file_path = os.path.join(parent_dir, os.path.basename(folder_path) + '.zip')

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Walk through each file and subdirectory in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Add the file to the zip archive with relative path
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    return zip_file_path

# if __name__ == "__main__":
#     folder_path = r"G:\MWaqar\Speech-to-Text\uploads\b"
#     create_zip_folder(folder_path)
