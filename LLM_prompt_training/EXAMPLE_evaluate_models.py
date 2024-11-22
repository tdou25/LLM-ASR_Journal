# twenty_five_percent_model.config.use_cache = False
# twenty_five_percent_model.eval()

tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from transformers import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(
    prompt,
    model,
    tokenizer,
    input=None,
    temperature=0.7,
    num_beams=5,
    max_new_tokens=128,
    **kwargs,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        num_beams=num_beams,
        **kwargs,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": False,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=1,
            no_repeat_ngram_size = 2

        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output.replace("<s>", "").replace("</s>", "")
    if "###Assistant" in output:
      output = output.split("###Assistant")[1]

    yield output


def find_model_predictions(prompts, model, tokenizer):
  predictions = []
  processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")

  for prompt in tqdm(prompts):
    for model_output in evaluate(prompt, model, tokenizer):
      model_output = processor.tokenizer._normalize(model_output)
      print(model_output)
      predictions.append(model_output)

  return predictions

original_test_other_preds = find_model_predictions(original_test_other_prompts, original_model, tokenizer)
original_test_other_wer = 100 * wer.compute(predictions=original_test_other_preds, references=original_test_other_refs)
original_test_other_cer = 100 * cer.compute(predictions=original_test_other_preds, references=original_test_other_refs)

results_file_path = "original_results.txt"
with open(results_file_path, "w") as results_file:
  results_file.write(f"LibriSpeech Test Other WER (original): {original_test_other_wer:.2f}%")
  results_file.write(f"LibriSpeech Test Other CER (original): {original_test_other_cer:.2f}%")