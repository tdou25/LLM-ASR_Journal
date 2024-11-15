import os
import whisper

if __name__ == "__main__":
    model = whisper.load_model("large")
    file_out = open("transcript.txt", 'w')
    
    result = model.transcribe("friends_romans_countrymen.mp3")

    file_out.write(result)