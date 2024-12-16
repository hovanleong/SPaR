import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int)
parser.add_argument('--end', type=int)
parser.add_argument('--gpu', type=int)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pandas as pd
from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer


# TODO: change data path, there should exist the key "prompt"
with open('', encoding='utf-8') as f:
    data = json.load(f)[args.begin: args.end]

tmp = [{'messages': [{'role': 'user', 'content': i['prompt']}]} for i in data]


# Create an LLM.
# TODO
model_path = 'Your-PATH-Here'

llm = LLM(model=model_path, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = "left"
prompts = []

for i in tmp:
    try:
        prompts.append(tokenizer.apply_chat_template(i['messages'], add_generation_prompt=True, tokenize=True))
    except Exception as e:
        print(e)
        continue    
print("data numbers: ", len(prompts))
print(tokenizer.decode(prompts[0]))

# stop_token_ids = [151329, 151336, 151338]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.9, 
    top_p=0.9,
    max_tokens=2048,
    # repetition_penalty=1.05,
    n=5,
    # stop_token_ids=stop_token_ids
    stop=['<|eot_id|>', '</s>']
)

# from IPython import embed; embed()

outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
# Print the outputs.
res = []
for output, i in zip(outputs, data):
    i['output'] = [output.outputs[j].text.strip() for j in range(5)]
    i['generator'] = 'llama3-8b-instruct'
    # i['generator'] = 'mistrial-7b-instruct'
    res.append(i)

with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
