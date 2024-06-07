import os
import json
from sklearn.model_selection import train_test_split

def load_dataset(dataset_dir):
    # Initialize an empty list to store samples
    samples = []

    # Assign unique numerical IDs to subfolders
    subfolder_ids = {subfolder: idx + 1 for idx, subfolder in enumerate(os.listdir(dataset_dir))}

    # Iterate through subfolders in the dataset directory
    for subfolder in os.listdir(dataset_dir):
        subfolder_path = os.path.join(dataset_dir, subfolder)

        # Ensure it's a directory
        if os.path.isdir(subfolder_path):
            # Process all files within the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)

                # Check if it's a file
                if os.path.isfile(file_path):
                    # Assuming audio and text files have the same base name in each subfolder
                    base_name, file_extension = os.path.splitext(file_name)

                    if file_extension == ".wav":
                        # Read audio content
                        with open(file_path, 'rb') as audio_file:
                            audio_content = audio_file.read()

                        # Read text content and remove newlines
                        text_file_path = os.path.join(subfolder_path, base_name + ".txt")
                        with open(text_file_path, 'r', encoding='utf-8') as text_file:
                            text_content = text_file.read().replace('\n', '')

                        # Modify the sample dictionary to include audio and text information
                        sample = {
                            "source": {
                                "id": subfolder_ids[subfolder],  # Assigning unique numerical IDs
                                "lang": "urd",
                                "text": text_content,  # Update based on your data
                                "audio_local_path": file_path,
                                "waveform": None,  # You may need to populate this field based on your data
                                "sampling_rate": 16000,  # Update with the actual sampling rate
                                "units": None  # You may need to populate this field based on your data
                            },
                            "target": {
                                "id": subfolder_ids[subfolder],  # Assigning unique numerical IDs
                                "lang": "urd",
                                "text": text_content,  # Update based on your data
                                "audio_local_path": file_path,
                                "waveform": None,  # You may need to populate this field based on your data
                                "sampling_rate": 16000,  # Update with the actual sampling rate
                                "units": None  # You may need to populate this field based on your data
                            }
                        }

                        # Append the sample to the list
                        samples.append(sample)

    return samples

def split_and_write_json(dataset_dir, output_dir, train_ratio=0.8):
    # Load dataset
    samples = load_dataset(dataset_dir)

    # Split dataset into training and validation sets
    train_samples, val_samples = train_test_split(samples, train_size=train_ratio, random_state=42)

    # Write training samples to JSON
    train_json_path = os.path.join(output_dir, "train.json")
    with open(train_json_path, 'w', encoding='utf-8') as train_json_file:
        for sample in train_samples:
            json.dump(sample, train_json_file, ensure_ascii=False)
            train_json_file.write('\n')  # Add a newline to separate entries

    # Write validation samples to JSON
    val_json_path = os.path.join(output_dir, "validation.json")
    with open(val_json_path, 'w', encoding='utf-8') as val_json_file:
        for sample in val_samples:
            json.dump(sample, val_json_file, ensure_ascii=False)
            val_json_file.write('\n')  # Add a newline to separate entries


# Example usage
dataset_directory = "/home/waqar/MWaqar/Speech-to-Text/P Dataset"
output_directory = "/home/waqar/MWaqar/Speech-to-Text/P Dataset"
split_and_write_json(dataset_directory, output_directory)
