import os
from datasets import Dataset, Audio, Value
import soundfile as sf
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
                    
                    
def save_dataset(dataset, output_dir, split_name):
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    dataset.save_to_disk(split_output_dir)

def prepare_data(source_data_dir, output_data_dir='op_data_dir'):
    with open(os.path.join(source_data_dir, 'audio_paths'), 'r', encoding='utf-8') as audio_entries_file:
        scp_entries = audio_entries_file.readlines()

    with open(os.path.join(source_data_dir, 'text'), 'r', encoding='utf-8') as text_entries_file:
        txt_entries = text_entries_file.readlines()

    if len(scp_entries) == len(txt_entries):
        audio_dataset = Dataset.from_dict({
            "audio": [audio_path.split(' ', 1)[1].strip() for audio_path in scp_entries],
            "sentence": [' '.join(text_line.split()[1:]).strip() for text_line in txt_entries]
        })

        audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        audio_dataset = audio_dataset.cast_column("sentence", Value("string"))

        # Split the dataset into training and validation
        train_ratio = 0.8  # Adjust the ratio as needed
        train_size = int(len(audio_dataset) * train_ratio)
        
        train_dataset = audio_dataset.select(list(range(train_size)))
        valid_dataset = audio_dataset.select(list(range(train_size, len(audio_dataset))))

        # Save training and validation datasets to disk
        save_dataset(train_dataset, output_data_dir, 'train')
        save_dataset(valid_dataset, output_data_dir, 'valid')

        print('Data preparation done. Training and validation datasets saved.')

    else:
        print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')


def print_sample_rates(root_dir):
    sample_rates = set()  # Initialize an empty set to store sample rates
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(subdir, file)
                try:
                    # Get the sample rate using the soundfile library
                    sample_rate = sf.info(audio_path).samplerate
                    sample_rates.add(sample_rate)  # Add sample rate to the set
                    break  # Stop processing files in this directory after the first one
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

    # Print unique sample rates
    if sample_rates:
        print("Unique Sample Rates:")
        for rate in sample_rates:
            print(rate)
    else:
        print("No audio files found.")
    return max(sample_rates)


def remove_empty_text_files(base_directory):
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".wav"):
                audio_file_path = os.path.join(root, file)
                text_file_path = os.path.join(root, file.replace(".wav", ".txt"))

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