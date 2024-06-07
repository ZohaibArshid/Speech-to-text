import os
import shutil
from pathlib import Path

def copy_files_to_folder(source_path="output_model_dir",checkpoint_path="checkpoint-38544", destination_folder="Tensor_Model"):
    # Check if the destination folder exists
    # if not os.path.exists(destination_folder):
        if os.path.exists(destination_folder):
             shutil.rmtree(destination_folder)
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder)
        checkpoint_path=os.path.join(source_path,checkpoint_path)
        # List of files to copy
        source_paths = [
            f"{source_path}/added_tokens.json",
            f"{source_path}/normalizer.json",
            f"{source_path}/preprocessor_config.json",
            f"{source_path}/special_tokens_map.json",
            f"{source_path}/tokenizer_config.json",
            f"{source_path}/merges.txt",
            f"{source_path}/vocab.json",
            f"{checkpoint_path}/config.json",
            f"{checkpoint_path}/model-00001-of-00002.safetensors",
            f"{checkpoint_path}/model-00002-of-00002.safetensors",
            f"{checkpoint_path}/model.safetensors.index.json",
            f"{checkpoint_path}/training_args.bin",
        ]

        # Copy files to the destination folder
        for file_path in source_paths:
            if os.path.exists(file_path):
                shutil.copy(file_path, destination_folder)
            else:
                print(f"Warning: File not found - {file_path}")
    # else:
    #     print(f"Destination folder already exists. Skipping copying process.")

# copy_files_to_folder(source_path="POUTPUT",checkpoint_path="checkpoint-22400", destination_folder="Tensor_Model")
copy_files_to_folder(source_path="chori_model",checkpoint_path="checkpoint-611", destination_folder="Tensor_Model")


# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
# import time

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# # Replace 'path_to_pretrained_model' with the actual path to your pre-trained model files
# path_to_pretrained_model = "Tensor_Model"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     path_to_pretrained_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(path_to_pretrained_model)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     # max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     torch_dtype=torch_dtype,
#     device=device,
# )



# def mp3_to_wav_without_loss(input_mp3, output_wav):
#     import subprocess
#     # Run ffmpeg command for conversion
#     subprocess.run(["ffmpeg", "-i", input_mp3, output_wav])

# # Example usage:
# audio_path = "AudioTestSample/headline.mp3"
# # output_wav_file = "AudioTestSample/nawaz.wav"

# # mp3_to_wav_without_loss(audio_path, output_wav_file)



# audio_path = audio_path
# start_time = time.time()
# result = pipe(audio_path)
# print("Pre-trained model")
# print(result["text"])
# print("Time taken is:", time.time() - start_time)


