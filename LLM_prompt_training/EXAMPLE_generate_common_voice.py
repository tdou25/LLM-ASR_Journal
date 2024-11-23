import os
import sys
from typing import List

import fire
import torch
import transformers
import numpy as np
from datasets import load_dataset, concatenate_datasets
from evaluate import load
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import DataCollatorForSeq2Seq, DataCollator

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb

accelerate launch --config_file config.yaml finetune.py 
hostname -i
netstat -an | grep LISTEN
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    PromptTuningConfig,
    TaskType,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter
# ['dense_4h_to_h', 'dense', 'lm_head', 'dense_h_to_4h', 'query_key_value']                                                      
MODEL_NAME = "whisper-small.en"
NUM_HYPOTHESES = 3

transformers.logging.set_verbosity_error()

# Get the entire librispeech dataset
train_clean_100_ds = load_dataset("librispeech_asr", "all", split="train.clean.100", cache_dir="datasets")
train_clean_360_ds = load_dataset("librispeech_asr", "all", split="train.clean.360", cache_dir="datasets")
train_other_500_ds = load_dataset("librispeech_asr", "all", split="train.other.500", cache_dir="datasets")

# Concatenate the datasets
data = concatenate_datasets([train_clean_100_ds, train_clean_360_ds, train_other_500_ds])
dev_data = load_dataset("librispeech_asr", "all", split="test.clean+test.other", cache_dir="datasets")

def train(
    use_lora: bool = True,
    use_prompt_tuning: bool = False,
    # model/data params
    base_model: str = "FreedomIntelligence/phoenix-inst-chat-7b",  # the only required argument
    # dev_data_path: str = f"input-data/{MODEL_NAME}/validation-other.json",
    output_dir: str = f"models/{MODEL_NAME}/all",
    val_set_size: int = 2864,
    # training hyperparams
    batch_size: int = 80, # 50 is good for 8 gpus, multiply by number of machines available
    micro_batch_size: int = 8, # 4 is standard
    num_epochs: int = 1,
    learning_rate: float = 2e-5,
    cutoff_len: int = 3225,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = 
       ["query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter
    # resume_from_checkpoint: str = f'./models/{MODEL_NAME}/all/checkpoint-epoch-2',
    prompt_template_name: str = "phoenix",  # The prompt template to use, will default to alpaca.
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    print("RANKS", local_rank, global_rank)
    if int(local_rank) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"use_lora: {use_lora}\n"
            f"base_model: {base_model}\n"
            #  f"dev_data_path: {dev_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model`: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='FreedomIntelligence/phoenix-inst-chat-7b'"
    # gradient_accumulation_steps = batch_size // micro_batch_size
    # print("TOTAL GRAD ACC STEPS", gradient_accumulation_steps)

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    # device_map={'':torch.cuda.current_device()}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": local_rank}
        gradient_accumulation_steps = batch_size // world_size
        print("WORLD SIZE", world_size, "NEW GRAD ACC STEPS", gradient_accumulation_steps)


    # Check if parameter passed or if set within environ
    use_wandb = False
    # use_wandb = len(wandb_project) > 0 or (
    #     "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    if use_lora or use_prompt_tuning:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map=device_map,
                cache_dir="cache/transformers",
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading the model: {e}")
            raise RuntimeError(f"Error loading the model: {e}, CUDA Available: {torch.cuda.is_available()}")
            # Add more details or logging if needed

    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map=device_map,
            cache_dir="cache/transformers",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        device_map=device_map,
        cache_dir="cache/transformers"
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        # print(tokenized_full_prompt)
        return tokenized_full_prompt

    if use_lora:
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif use_prompt_tuning:
        model = prepare_model_for_kbit_training(model)

        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=8,
            tokenizer_name_or_path=tokenizer,
        )
        model = get_peft_model(model, config)
    
    # if dev_data_path.endswith(".json") or dev_data_path.endswith(".jsonl"):
    #     dev_data = load_dataset("json", data_files=dev_data_path, cache_dir="cache/datasets")
    # else:
    #     dev_data = load_dataset(dev_data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            # resume_from_checkpoint = (
            #     False  # So the trainer won't try loading its state
            # )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if use_lora:
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.


    if val_set_size > 0:
        train_data = data.shuffle()
        val_data = dev_data.shuffle()
        print("VAL DATA SIZE:", len(val_data))
        # val_data = None
        # val_data = dev_data['train'].shuffle().map(generate_and_tokenize_prompt, load_from_cache_file=True)
        # print("VAL DATA", val_data[0])
    else:
        train_data = data.shuffle()
        val_data = None

    # if not ddp and torch.cuda.device_count() > 1:
    #     print("DDP 2!!!")
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     device_map = {"": torch.cuda.current_device()}

    #     model.is_parallelizable = True
    #     model.model_parallel = True

    # Get the whisper model and tokenizer
    asr_model = WhisperForConditionalGeneration.from_pretrained(f"openai/{MODEL_NAME}", cache_dir="cache/transformers")
    asr_processor = WhisperProcessor.from_pretrained(f"openai/{MODEL_NAME}", cache_dir="cache/transformers")

    seq2seq_data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    class CompositeDataCollator:
        def __init__(self, seq2seq_data_collator):
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(f"openai/{MODEL_NAME}", cache_dir="cache/transformers").to(f"cuda:{local_rank}")
            self.asr_processor = WhisperProcessor.from_pretrained(f"openai/{MODEL_NAME}", cache_dir="cache/transformers")
            self.seq2seq_data_collator = seq2seq_data_collator

        def __call__(self, examples):
            if not examples:
                return []
            instruction = f"Perform error correction on the top {NUM_HYPOTHESES} outputs generated by an Automatic Speech Recognition (ASR) system. The ASR hypotheses, listed in order of their ASR posterior score, are as follows:\n"

            data = []
            for example in examples:
                if "instruction" in example and "input" in example and "output" in example:
                    print("ERROR: Skipping already processed example:", example)
                    continue

                audio = example["audio"]
                input_features = self.asr_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
                with torch.no_grad():
                    beam_outputs = self.asr_model.generate(
                        input_features.to(f"cuda:{local_rank}"),
                        num_beams=NUM_HYPOTHESES,
                        num_return_sequences=NUM_HYPOTHESES,
                        max_new_tokens=256,
                        early_stopping=True
                    )
                
                input_text = ""
                for i, beam_output in enumerate(beam_outputs):
                    hypothesis = f"<hypothesis{i + 1}>" + self.asr_processor.tokenizer._normalize(self.asr_processor.decode(beam_output, skip_special_tokens=True)) + f"</hypothesis{i + 1}>\n"
                    input_text += hypothesis

                end_instruction = "Please provide the corrected top1 ASR transcription of the given utterance only, do not add any explanations or other words.\n"
                input_text += end_instruction
                output = self.asr_processor.tokenizer._normalize(example["text"])

                data_point = {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                }
                data.append(data_point)

            # Tokenize all of the data points
            data = [generate_and_tokenize_prompt(data_point) for data_point in data]

            # Apply the Seq2Seq data collator to the new batch
            seq2seq_batch = self.seq2seq_data_collator(data)
            return seq2seq_batch
    
    composite_data_collator = CompositeDataCollator(seq2seq_data_collator)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size // world_size,
            per_device_eval_batch_size=batch_size // world_size,
            # eval_accumulation_steps = 1,
            gradient_accumulation_steps=micro_batch_size,
            # warmup_steps=len(train_data) // 10, # Rougly 10% of the training data
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            optim="adamw_torch",
            logging_steps=5,
            logging_strategy="steps",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # eval_steps=5 ,
            # save_steps=5,
            logging_first_step = True,
            output_dir=output_dir,
            save_total_limit=4,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            remove_unused_columns=False,
            # metric_for_best_model="wer",
            # greater_is_better=False,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=composite_data_collator,
        # compute_metrics=compute_metrics
    )
    model.config.use_cache = False
    # early_stopping_callback = transformers.EarlyStoppingCallback(
    #     early_stopping_patience=4,
    #     early_stopping_threshold=1e-3,
    # )
    # trainer.add_callback(early_stopping_callback)

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
    trainer.train()
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)