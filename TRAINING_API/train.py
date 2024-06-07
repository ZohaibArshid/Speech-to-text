from utils import create_dataset,prepare_data,print_sample_rates,remove_empty_text_files
from config import *
from helper import train_whisper_model
##########################
#remove empty audio and text file
remove_empty_text_files(dataset_path)
##########################
# sampling Rate
sample_rate=print_sample_rates(dataset_path)

# ##########################
# ##########################
# # Path generation
# create_dataset(dataset_path,audio_text_path)
# # ###########################
# # #Dataset Split
# prepare_data(audio_text_path,dataset_split)
###########################
# Training
train_whisper_model(sampling_rate=sample_rate,
                    train_datasets=train_datasets,
                    eval_datasets=eval_datasets,
                    output_dir=output_dir)
###########################