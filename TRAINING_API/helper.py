import os 
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate

def train_whisper_model(model_name='openai/whisper-large-v2',
                        language='Urdu',
                        sampling_rate=16000,
                        num_proc=1,
                        train_strategy='epoch',
                        learning_rate=5e-6,
                        warmup=20000,
                        train_batchsize=16,
                        eval_batchsize=8,
                        num_epochs=10000,
                        num_steps=100000,
                        resume_from_ckpt='',
                        output_dir='output_model_dir',
                        train_datasets="",
                        eval_datasets=""):
    
    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
    print('PARAMETERS OF INTEREST:')
    print({
        'model_name': model_name,
        'language': language,
        'sampling_rate': sampling_rate,
        'num_proc': num_proc,
        'train_strategy': train_strategy,
        'learning_rate': learning_rate,
        'warmup': warmup,
        'train_batchsize': train_batchsize,
        'eval_batchsize': eval_batchsize,
        'num_epochs': num_epochs,
        'num_steps': num_steps,
        'resume_from_ckpt': resume_from_ckpt,
        'output_dir': output_dir,
        'train_datasets': train_datasets,
        'eval_datasets': eval_datasets
    })
    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

    gradient_checkpointing = True
    freeze_feature_encoder = False
    freeze_encoder = False

    do_normalize_eval = True
    do_lower_case = False
    do_remove_punctuation = False
    normalizer = BasicTextNormalizer()

    #############################       MODEL LOADING       #####################################

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    if freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if gradient_checkpointing:
        model.config.use_cache = False

    ############################        DATASET LOADING AND PREP        ##########################

    def load_custom_dataset(split):
        ds = []
        if split == 'train':
            for dset in train_datasets:
                ds.append(load_from_disk(dset))
        if split == 'eval':
            for dset in eval_datasets:
                ds.append(load_from_disk(dset))

        ds_to_return = concatenate_datasets(ds)
        ds_to_return = ds_to_return.shuffle(seed=22)
        return ds_to_return

    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        transcription = batch["sentence"]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = normalizer(transcription).strip()
        
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    max_label_length = model.config.max_length
    min_input_length = 0.0
    max_input_length = 30.0
    def is_in_length_range(length, labels):
        return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

    print('DATASET PREPARATION IN PROGRESS...')
    raw_dataset = DatasetDict()
    raw_dataset["train"] = load_custom_dataset('train')
    raw_dataset["eval"] = load_custom_dataset('eval')

    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    raw_dataset = raw_dataset.map(prepare_dataset, num_proc=num_proc)

    raw_dataset = raw_dataset.filter(
        is_in_length_range,
        input_columns=["input_length", "labels"],
        num_proc=num_proc,
    )

    ###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    print('DATASET PREPARATION COMPLETED')

    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    ###############################     TRAINING ARGS AND TRAINING      ############################

    if train_strategy == 'epoch':
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batchsize,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_steps=warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=num_epochs,
            save_total_limit=10,
            per_device_eval_batch_size=eval_batchsize,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=500,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            optim="adamw_bnb_8bit",
            resume_from_checkpoint=resume_from_ckpt,
        )

    elif train_strategy == 'steps':
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_batchsize,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            warmup_steps=warmup,
            gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            max_steps=num_steps,
            save_total_limit=10,
            per_device_eval_batch_size=eval_batchsize,
            predict_with_generate=True,
            generation_max_length=225,
            logging_steps=500,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            optim="adamw_bnb_8bit",
            resume_from_checkpoint=resume_from_ckpt,
        )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    print('TRAINING IN PROGRESS...')
    trainer.train()
    print('DONE TRAINING')


# train_whisper_model()
