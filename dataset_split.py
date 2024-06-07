import os
import argparse
from datasets import Dataset, Audio, Value

def save_dataset(dataset, output_dir, split_name):
    split_output_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    dataset.save_to_disk(split_output_dir)

parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
parser.add_argument('--source_data_dir', type=str, required=True, help='Path to the directory containing the audio_paths and text files.')
parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')

args = parser.parse_args()

with open(os.path.join(args.source_data_dir, 'audio_paths'), 'r', encoding='utf-8') as audio_entries_file:
    scp_entries = audio_entries_file.readlines()

with open(os.path.join(args.source_data_dir, 'text'), 'r', encoding='utf-8') as text_entries_file:
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
    save_dataset(train_dataset, args.output_data_dir, 'train')
    save_dataset(valid_dataset, args.output_data_dir, 'valid')

    print('Data preparation done. Training and validation datasets saved.')

else:
    print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')


# python .\custom_data\dataset_split.py --source_data_dir .\custom_data\sample_data\ --output_data_dir DatasetSplit
# python .\train\fine-tune_on_custom_dataset.py --train_datasets "DatasetSplit/train" --eval_datasets "DatasetSplit/valid"