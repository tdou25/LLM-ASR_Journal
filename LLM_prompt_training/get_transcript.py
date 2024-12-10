import os
import whisper

if __name__ == "__main__":
    model = whisper.load_model("small")
    file_out = open("transcript.txt", 'w')
    
    result = model.transcribe("audio_files/Genesis_01.mp3")

    print(result)

    file_out.write(result)