from datasets import load_dataset

from huggingface_hub import login

key = open("read_tok.txt")
token = key.read()

login(token)

dataset = load_dataset("oscar-corpus/OSCAR-2301",
                        language="am", 
                        streaming=True, 
                        split="train") 

iterations = 0

for d in dataset:
    if iterations > 10:
        break

    print(d) 
    iterations += 1

