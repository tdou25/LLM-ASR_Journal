from datasets import load_dataset
from datasets import Audio

from huggingface_hub import login

key = open("read_tok.txt")
token = key.read()

login(token)

dataset = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
                        # language="am", 
                        # streaming=True, 
                        # split="train") 

iterations = 0

#print(dataset["train"][0])

gigaspeech = dataset.cast_column("audio", Audio(sampling_rate=8000))

# for d in dataset:
#     if iterations > 10:
#         break

#     print(d) 
#     iterations += 1

