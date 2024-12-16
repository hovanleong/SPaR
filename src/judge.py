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


# TODO: change data file
with open('', encoding='utf-8') as f:
    data = json.load(f)[args.begin: args.end]


def build_judge_template(prompt, response):
    return """Please act an expert in evaluating the capabilities of instruction-following. In the instruction-following task, the Output needs to honestly/precisely/closely follow the given Prompt.
Your task is to carefully judge whether the Output to honestly/precisely/closely follows the given Prompt. If there are any constraints in the Prompt that are not satisfied by the Output, please list all the constraints that are not satisfied.

Prompt: “{}”

Output: “{}”

Please carefully judge if each constraint is perfectly satisfied and give a final judgement weather the Output accurately follows the Prompt in the following format:
Step-by-step verification: xxx
Final Judgement (if the Output accurately follows the Prompt): (Yes or No)""".format(prompt, response)


# TODO: check keys
tmp = [{'messages': [{'role': 'user', 'content': build_judge_template(i['prompt'], i['response'])}]} for i in data]

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
    temperature=0.7, 
    top_p=1.0,
    max_tokens=2048,
    # repetition_penalty=1.05,
    n=5,
    # n=1,
    # stop_token_ids=stop_token_ids
    stop=['<|eot_id|>', '</s>']
)


outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sampling_params)


res = []
for output, i in zip(outputs, data):
    i['critique'] = [output.outputs[j].text.strip() for j in range(5)]
    
    i['generator'] = 'llama3-8b-instruct'
    res.append(i)

with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)
