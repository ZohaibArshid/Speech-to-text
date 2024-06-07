# from faster_whisper import WhisperModel
# import time
# start_time=time.time()
# model_size = "large-v2"
<<<<<<< HEAD
audio_path=r"AudioTestSample/y2mate.bz - was-offered-usd-5-billion-by-bill-clinton-not-to-conduct-nuclear-tests-nawaz-sharif.mp3"
# # # Run on GPU with FP16
=======
# audio_path=r"AudioTestSample/President_address_National_Calligraphy_Exhibition.wav"
# # Run on GPU with FP16
>>>>>>> 15d8ac37922f1d339a9783805fb77a750911fffd
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# # or run on GPU with INT8
# # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
# # model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = model.transcribe(audio_path, beam_size=5)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
# print("faster whisper ")
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
# print("time taking is:",time.time()-start_time)

# import whisper
# start_time=time.time()
# model = whisper.load_model("large-v2")

# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio(audio_path)
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print("whisper ")
# print(result.text)
# print("time taking is:",time.time()-start_time)
  
# start_time=time.time()
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v2"
# # model_id="temp_dir"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# result = pipe(audio_path,generate_kwargs={"language": "urdu", "task": "transcribe"})
# print("whisper with pipe")
# print(result["text"])
# print("time taking is:",time.time()-start_time)


# model_id = "openai/whisper-large-v3"
# # model_id="temp_dir"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     max_new_tokens=128,
#     chunk_length_s=30,
#     batch_size=16,
#     return_timestamps=True,
#     torch_dtype=torch_dtype,
#     device=device,
# )

# result = pipe(audio_path,generate_kwargs={"language": "urdu", "task": "transcribe"})
# print("whisper v3 with pipe")
# print(result["text"])
# print("time taking is:",time.time()-start_time)


# start_time=time.time()
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
# from utils import split_audio
from utils import split_on_silence
from utils import split_audio
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v2"
model_id="Tensor_Model"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    # max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    # return_timestamps=True,
    # no_timestamps_token_id=True,
    torch_dtype=torch_dtype,
    device=device,
)
# for audio_path in split_on_silence(audio_path):
for audio_path in split_audio("/home/waqar/MWaqar/Speech-to-Text/chori/datatset/chori1.wav"):
    result = pipe(audio_path)
    # print("own model")
    print(result["text"])


