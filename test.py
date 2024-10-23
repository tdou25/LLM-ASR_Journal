import os
import whisper

if __name__ == "__main__":
    model = whisper.load_model("base")
    
    # Expand the file path
    file_path = os.path.expanduser("/Users/tdou/Documents/RCOS_workspace/ASR_Enhancement_LLM/LibriSpeech/test-clean/61/70968/61-70968-0000.flac")
    
    result = model.transcribe(file_path)
    print(f' The text in video: \n {result["text"]}')