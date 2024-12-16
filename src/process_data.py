import json
import random
import re


def process_gen_res(input_path, output_path):
    d = []
    for i in range(8):
        with open(f'{input_path}/vllm_output_{i}.json', encoding='utf-8') as f:
            d.extend(json.load(f))

    num = 0
    res = []
    for i in d:
        for j in range(5):
            tmp = i.copy()
            tmp['id'] = '{}_{}'.format(num, j)
            tmp['response'] = i['output'][j]
            res.append(tmp)
        num += 1
    
    print(res[:6])
    print(len(res))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)


def get_acc(resp):
    if 'Final Judgement' not in resp:
        return -1
    resp = resp.split('Final Judgement')[1].strip()
    if not resp.count('Yes') and not resp.count('No'):
        return -1
    if resp.count("Yes"):
        if resp.find("No") != -1 and resp.find("No") < resp.find("Yes"):
            return  0
        else:
            return  1
    elif resp.count("No"):
        if resp.find("Yes") != -1 and resp.find("Yes") < resp.find("No"):
            return  1
        else:
            return  0



def process_score_res_voting(input_path, output_path):
    l = []
    for i in range(8):
        with open(f'{input_path}/vllm_output_{i}.json', encoding='utf-8') as f:
            l.extend(json.load(f))

    
    d = []
    for i in l:
        tmp = []
        for j in i['critique']:
            tmp.append(get_acc(j))
        
        if tmp.count(0) == tmp.count(1):
            continue
        elif tmp.count(0) > tmp.count(1):
            i['acc'] = 0
        else:
            i['acc'] = 1
        
        for j in range(len(tmp)):
            if tmp[j] == i['acc']:
                i['critique'] = i['critique'][j]
                break
        
        d.append(i)
    
    res = {}
    for i in d:
        prompt_id = int(i['id'].split('_')[0])
        res_id = int(i['id'].split('_')[1])
        if prompt_id not in res:
            res[prompt_id] = {
                'original_prompt': i['original_prompt'],
                'prompt': i['prompt'],
                'output': [],
                'acc': [],
                'critique': [],
                'prompt_id': prompt_id
            }
        res[prompt_id]['output'].append(i['output'][res_id])
        res[prompt_id]['acc'].append(i['acc'])
        res[prompt_id]['critique'].append(i['critique'])
    
    rft_data = []
    for i in res:
        if min(res[i]['acc']) == 0:
            rft_data.append(res[i])
    
    bad_data = []
    for i in rft_data:
        tmp = []
        for j in range(len(i['acc'])):
            if i['acc'][j] == 0:
                tmp.append(j)
        random.shuffle(tmp)
        for j in tmp[:1]:
            bad_data.append({
                'prompt': i['prompt'],
                'response': i['output'][j],
                'critique': i['critique'][j],
                'acc': 0,
                'prompt_id': i['prompt_id']
            })
    print(len(bad_data))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bad_data, f, indent=4, ensure_ascii=False)


def process_dpo_data(input_path, output_path):
    
    with open(input_path, encoding='utf-8') as f:
        l = f.readlines()
    
    d = []
    for i in l:
        i = json.loads(i)
        if i['final_node']['value'] > 0.5:
            d.append(i)
    
    data = []
    for i in d:
        data.append({
            'messages': [
                {
                    "role": "user",
                    "content": i['prompt']
                }
            ],
            "chosen": {
                "role": "assistant",
                "content": i['final_node']['response']
            },
            "rejected": {
                "role": "assistant",
                "content": i['response']
            }
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

judge_template = """Please act an expert in evaluating the capabilities of instruction-following. In the instruction-following task, the Output needs to honestly/precisely/closely follow the given Prompt.
Your task is to carefully judge whether the Output to honestly/precisely/closely follows the given Prompt. If there are any constraints in the Prompt that are not satisfied by the Output, please list all the constraints that are not satisfied.

Prompt: “{}”

Output: “{}”

Please carefully judge if each constraint is perfectly satisfied and give a final judgement weather the Output accurately follows the Prompt in the following format:
Step-by-step verification: xxx
Final Judgement (if the Output accurately follows the Prompt): (Yes or No)"""

def process_rft_data(input_tree_search_path, input_judge_path, output_path):
    
    with open(input_tree_search_path, encoding='utf-8') as f:
        l = f.readlines()
    refine_data = []
    for i in l:
        i = json.loads(i)
        if i['final_node']['value'] > 0.5:
            refine_data.append({
                'messages': [
                    {"role": "user", "content": judge_template.format(i['prompt'], i['final_node']['parent_res'])},
                    {"role": "assistant", "content": i['final_node']['parent_critique']},
                    {"role": "user", "content": """Based on your judgement, refine the Output to make sure it can honestly/precisely/closely follows the given Prompt.


    Please carefully refine the Output to meet all the constraints in the Prompt. 

    Please format like this:
    Reflection on how to refine the Output: xxx
    Final Refined Output: [[start]] xxx [[end]]"""},
                    {
                        "role": "assistant",
                        "content": i['final_node']['refine_cot']
                    }
                ]
            })
    pos = []
    neg = []
    for i in l:
        i = json.loads(i)
        if len(i['critique_rft']['positive']):
            pos.append({
                'messages': [
                    {"role": "user", "content": i['critique_rft']['positive'][0]['prompt']},
                    {"role": "assistant", "content": i['critique_rft']['positive'][0]['response']}
                ]
            })
        if len(i['critique_rft']['negative']):
            neg.append({
                'messages': [
                    {"role": "user", "content": i['critique_rft']['negative'][0]['prompt']},
                    {"role": "assistant", "content": i['critique_rft']['negative'][0]['response']}
                ]
            })
    l = []
    for i in range(8):
        with open(f'{input_judge_path}/vllm_output_{i}.json', encoding='utf-8') as f:
            l.extend(json.load(f))

    d_0 = []
    d_1 = []
    for i in l:

        tmp = []
        for j in i['critique']:
            tmp.append(get_acc(j))
        
        if tmp.count(0) == tmp.count(1):
            continue
        elif tmp.count(0) > tmp.count(1):
            i['acc'] = 0
        else:
            i['acc'] = 1
        
        good_res = None
        bad_res = None
        for j in range(len(tmp)):
            if good_res and bad_res:
                break
            if tmp[j] == i['acc']:
                good_res = i['critique'][j]
            elif tmp[j] == 1-i['acc']:
                bad_res = i['critique'][j]
        if not (good_res and bad_res):
            continue
        i['good_res'] = good_res
        i['bad_res'] = bad_res    
        i['judge_prompt'] = judge_template.format(i['prompt'], i['response'])
        if i['acc'] == 0:
            d_0.append(i)
        else:
            d_1.append(i) 
    random.shuffle(pos)
    random.shuffle(neg)
    random.shuffle(d_0)
    random.shuffle(d_1)
    random.shuffle(refine_data)
    
    sft_data = []
    p = []
    for i in refine_data[:4000]:
        if i['messages'][0]['content'] not in p:
            p.append(i['messages'][0]['content'])
            sft_data.append(i)
    for i in pos[:1500]:
        if i['messages'][0]['content'] not in p:
            p.append(i['messages'][0]['content'])
            sft_data.append(i)
    for i in neg[:1000]:
        if i['messages'][0]['content'] not in p:
            p.append(i['messages'][0]['content'])
            sft_data.append(i)
    for i in d_0[:500]:
        if i['judge_prompt'] not in p:
            p.append(i['judge_prompt'])
            sft_data.append({
                'messages': [
                    {"role": "user", "content": i['judge_prompt']},
                    {"role": "assistant", "content": i['good_res']}
                ]
            })
    for i in d_1[:2500]:
        if i['judge_prompt'] not in p:
            p.append(i['judge_prompt'])
            sft_data.append({
                'messages': [
                    {"role": "user", "content": i['judge_prompt']},
                    {"role": "assistant", "content": i['good_res']}
                ]
            })
    
    random.shuffle(sft_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    input_path = ''
    output_path = ''
    process_gen_res(input_path, output_path)
    
    
    input_path = ''
    output_path = ''
    process_score_res_voting(input_path, output_path)
    
    
    input_path = ''
    output_path = ''
    process_dpo_data(input_path, output_path)
    
    
    input_tree_search_path = ''
    input_judge_path = ''
    output_path = ''
    process_rft_data(input_tree_search_path, input_judge_path, output_path)