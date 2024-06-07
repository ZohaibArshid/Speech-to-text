# importing libraries 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence


# a function that splits the audio file into chunks on silence
# and applies speech recognition
def get_large_audio_chunks_on_silence(path,folder_name):

    # open the audio file using pydub
    sound = AudioSegment.from_file(path)  
    # split audio sound where silence is 500 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 1000,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    chunks_path = []

    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        chunks_path.append(chunk_filename)

    return  chunks_path
# print(get_large_audio_chunks_on_silence(r"G:\MWaqar\Speech-to-Text\uploads\a\a.wav",r"G:\MWaqar\Speech-to-Text\uploads\a\chunks"))