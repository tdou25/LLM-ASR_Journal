from datasets import load_dataset

dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
for sample in dataset:
    print(sample["audio"]["array"])