from datasets import load_dataset
import os
import shutil

def copy_audio_files(dataset, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    for split_name in dataset.keys():
        split_data = dataset[split_name]
        for i, example in enumerate(split_data):
            audio_path = os.path.join(output_directory, f"audio_{i}.wav")
            text_path = os.path.join(output_directory, f"audio_{i}.txt")

            # Copy audio file
            shutil.copy(example['path'], audio_path)

            # Save transcription to text file
            with open(text_path, 'w', encoding='utf-8') as text_file:
                text_file.write(example['sentence'])

if __name__ == "__main__":
    # Replace with your actual dataset name and language abbreviation
    dataset_name = "mozilla-foundation/common_voice_13_0"
    language_abbr = "ur"

    download_directory = r"common_voice"
    cache_directory = r"common_voice"

    common_voice = load_dataset(
        dataset_name,
        language_abbr,
        data_dir=download_directory,
        cache_dir=cache_directory,
        use_auth_token=True,
    )

    output_directory = r"Dataset/Custom_dataset/common_voice"
    copy_audio_files(common_voice, output_directory)

    print("Audio files copied in:", output_directory)