import os 
##########################
dataset_path = os.path.join("..", "Dataset", "Custom_dataset")
audio_text_path = os.path.join("..", "Dataset")
##########################
dataset_split = os.path.join("..", "Split_Dataset")
##########################
model_name = 'openai/whisper-large-v2'
language = 'Urdu'
sampling_rate = 16000
num_proc = 1
train_strategy = 'epoch'
learning_rate = 5e-6
warmup = 20000
train_batchsize = 16
eval_batchsize = 8
num_epochs = 10000
num_steps = 100000
resume_from_ckpt = ''
output_dir = 'model_checkpoint'
train_datasets = os.path.join(dataset_split,"train")
eval_datasets = os.path.join(dataset_split,"valid")
