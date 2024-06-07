from datasets import load_dataset
import os
import shutil

def copy_audio_files(dataset, output_directory):


    for split_name in dataset.keys():
        split_data = dataset[split_name]
        # print(split_name)
        # print(split_data)
        for i, example in enumerate(split_data):
            audio_path = os.path.join(output_directory, f"{split_name}_audio_{i}.wav")
            text_path = os.path.join(output_directory, f"{split_name}_audio_{i}.txt")
            directory_path, file_name=os.path.split(example['path'])
            # print("Audio Path:", example['path'])
            # print(directory_path, file_name)
            
            # print("Text Path:", text_path)
            if split_name=="validation":
                file_audio_path=os.path.join(directory_path,"dev",file_name)
            else:
                file_audio_path=os.path.join(directory_path,split_name,file_name)

            # Check if the file exists before attempting to copy
            if os.path.exists(file_audio_path):
                # Attempt to copy the audio file
                try:
                    shutil.copy(file_audio_path, audio_path)
                    # print("Audio file copied successfully.")
                except Exception as e:
                    print(f"Error copying audio file: {e}")

                # Save transcription to text file
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(example['transcription'])
                    # print("Text file created successfully.")
            else:
                print("Warning: Audio file does not exist.")

if __name__ == "__main__":
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    output_directory = r"Dataset/Custom_dataset/fleurs"

    # Remove the entire directory if it exists
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
        print(f"Folder '{output_directory}' removed.")
        # time.sleep(1)

    # Create the directory
    os.makedirs(output_directory, exist_ok=True)
    print(f"Folder '{output_directory}' created.")
    # Replace with your actual dataset name and language abbreviation
    dataset_name = "google/fleurs"
    language_abbr = "ur_pk"
# fleurs_retrieval = load_dataset("google/fleurs", "ur_pk")
    download_directory = r"fleurs"
    cache_directory = r"fleurs"

    common_voice = load_dataset(
        dataset_name,
        language_abbr,
        data_dir=download_directory,
        cache_dir=cache_directory,
        use_auth_token=True,
    )
    # print(common_voice)

    copy_audio_files(common_voice, output_directory)

    print("Audio files copied in:", output_directory)