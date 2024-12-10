import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load

# Constants
MODEL_NAME = "whisper-small.en"
PHOENIX_MODEL = "FreedomIntelligence/phoenix-inst-chat-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JSON_DATASET_PATH = "/scratch-shared/historical-english/your_dataset.json"  # Path to your JSON file
OUTPUT_DIR = "/scratch-shared/historical-english/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON dataset


def load_json_dataset(json_path):
    """
    Load dataset from a JSON file.
    The file should contain a list of objects with `audio_path` and `text`.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Generate Whisper ASR outputs


def generate_asr_outputs(dataset, whisper_model, processor):
    """
    Generate ASR outputs for a dataset using Whisper.
    """
    asr_outputs = []
    for example in tqdm(dataset):
        try:
            audio_input = processor(
                example["audio_path"], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(DEVICE)

            with torch.no_grad():
                output_ids = whisper_model.generate(
                    audio_input, max_length=256)

            transcription = processor.batch_decode(
                output_ids, skip_special_tokens=True)[0]
            asr_outputs.append({
                "audio_path": example["audio_path"],
                "reference": example["text"],
                "asr_output": transcription,
            })

        except Exception as e:
            print(f"Error processing {example['audio_path']}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    return asr_outputs

# Format prompts for Phoenix


def format_for_phoenix(asr_outputs):
    """
    Prepare ASR outputs and references as input-output pairs for Phoenix zero-shot learning.
    """
    phoenix_inputs = []
    for item in asr_outputs:
        input_text = (
            f"ASR Hypothesis: {item['asr_output']}\n"
            f"Reference Transcription: {item['reference']}\n"
            f"Please provide the corrected transcription:"
        )
        phoenix_inputs.append(
            {"input": input_text, "output": item["reference"]})
    return phoenix_inputs

# Save formatted data


def save_as_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # Load models
    whisper_processor = WhisperProcessor.from_pretrained(
        f"openai/{MODEL_NAME}", cache_dir="cache/transformers")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        f"openai/{MODEL_NAME}", cache_dir="cache/transformers").to(DEVICE)

    phoenix_tokenizer = AutoTokenizer.from_pretrained(
        PHOENIX_MODEL, cache_dir="cache/transformers")
    phoenix_model = AutoModelForCausalLM.from_pretrained(
        PHOENIX_MODEL, cache_dir="cache/transformers").to(DEVICE)

    # Load dataset
    dataset = load_json_dataset(JSON_DATASET_PATH)
    print(f"Loaded dataset with {len(dataset)} examples.")

    # Generate ASR outputs
    asr_outputs = generate_asr_outputs(
        dataset, whisper_model, whisper_processor)
    save_as_json(asr_outputs, os.path.join(
        OUTPUT_DIR, "whisper_asr_outputs.json"))
    print(f"ASR outputs saved to {OUTPUT_DIR}/whisper_asr_outputs.json")

    # Format data for Phoenix
    phoenix_data = format_for_phoenix(asr_outputs)
    save_as_json(phoenix_data, os.path.join(OUTPUT_DIR, "phoenix_input.json"))
    print(f"Formatted Phoenix input saved to {OUTPUT_DIR}/phoenix_input.json")

    # Zero-shot evaluation with Phoenix
    phoenix_model.eval()
    results = []
    for item in tqdm(phoenix_data):
        inputs = phoenix_tokenizer(
            item["input"], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = phoenix_model.generate(**inputs, max_length=256)
        corrected_output = phoenix_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        results.append({
            "input": item["input"],
            "expected_output": item["output"],
            "model_output": corrected_output,
        })

    # Save Phoenix outputs
    save_as_json(results, os.path.join(OUTPUT_DIR, "phoenix_outputs.json"))
    print(f"Phoenix outputs saved to {OUTPUT_DIR}/phoenix_outputs.json")


if __name__ == "__main__":
    main()