import torch
from seamless_communication.inference import Translator


# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("finetune", "vocoder_36langs", torch.device("cuda:0"), torch.float16)