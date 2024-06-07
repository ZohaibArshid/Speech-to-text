# Speech to Text


## Getting Started

Follow the steps below to get started with this project:

## Using This Repository
### Environment
* Python 3.11.3
* Cuda 11.2
* Torch 2.0.1+cu118
### Installation
1. Clone the repository
```
git clone https://github.com/MuhammadWaqar621/Speech-to-Text.git
```

2. Install the requirements
```
conda create -n whisper_finetune python=3.11.4
conda activate whisper_finetune
pip install -r requirements.txt
```

3. Download Common Voice Dataset Dataset
```
python download_dataset.py
```

4. Prepare Dataset
```
    Dataset/
    │
    ├── Custom_dataset/
    │   ├── common_voice/
    │   │   ├── audio1.wav
    │   │   ├── audio1.txt
    │   │   └── ...
    │   ├── folder1/
    │   │   ├── audio1.wav
    │   │   ├── audio1.txt
    │   │   └── ...
    |   │── ├── folder2/
    │   │   ├── audio1.wav
    │   │   ├── audio1.txt
    │   │   └── ...
    
    
```
5. Generate audio_paths and text file
```
python GenerateDataPaths.py
```
6. Split Dataset
```
python dataset_split.py --source_data_dir Dataset --output_data_dir DatasetSplit
```
7. FineTune
```
python .\train\fine-tune_on_custom_dataset.py --train_datasets "DatasetSplit/train" --eval_datasets "DatasetSplit/valid"

```






