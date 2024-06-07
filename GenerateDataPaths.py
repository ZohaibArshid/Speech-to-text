import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    return content

def create_dataset(dataset_path, output_path):
    audio_path_file = os.path.join(output_path, "audio_paths")
    text_file = os.path.join(output_path, "text")

    with open(audio_path_file, 'w', encoding='utf-8') as audio_path_fp, open(text_file, 'w', encoding='utf-8') as text_fp:
        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)

            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                audio_written = False
                text_written = False
                
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)

                    # Check if it's an audio file
                    if file.endswith(".wav"):
                        utt_id = file.split('.')[0]  # Assuming the file name is the utt_id
                        text_file_path = os.path.join(subfolder_path, f"{utt_id}.txt")
                        
                        # Check if both audio and text files exist
                        if os.path.exists(text_file_path):
                            transcript = read_file(text_file_path)
                            text_fp.write(f"{utt_id} {transcript}\n")
                            text_written = True

                            audio_path_fp.write(f"{utt_id} {file_path}\n")
                            audio_written = True

                # If both audio and text files exist, write to the files
                if audio_written and text_written:
                    print(f"Processed folder: {subfolder}")

if __name__ == "__main__":
    dataset_path = os.path.join("Dataset","Custom_dataset")
    output_path = r"Dataset"
    # dataset_path = r"P Dataset"
    # output_path = r"P Dataset"
    # dataset_path = r"chori"
    # output_path = r"chori"
    create_dataset(dataset_path, output_path)
