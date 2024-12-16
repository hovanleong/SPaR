import requests
import multiprocessing
from multiprocessing import Manager
import json
from tqdm import tqdm
import os
import time
# import pandas as pd
import random
from ToT.task import *




def chat_gpt(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            # TODO: change model path, it should be same as the name deployed with vllm
            task = ToT_Task(m['prompt'], m['response'], m['critique'], root_value=0, propose_method='llama', algorithm='bfs', refine_model_path='', critique_model_path='', openai_api_base=m['openai_api_base'])
            solution, root, final_node = task.run()
            m['final_node'] = {
                'response': final_node.response,
                'refine_cot': final_node.refine_cot,
                'parent_res': final_node.parent.response,
                'parent_critique': final_node.parent.critique,
                'value': final_node.V,
            }
            negative_critique, positive_critique = root.getCritiqueRFT()
            if len(negative_critique):
                negative_critique = random.sample(negative_critique, 1)
            if len(positive_critique):
                positive_critique = random.sample(positive_critique, 1)
            m['critique_rft'] = {
                'positive': positive_critique,
                'negative': negative_critique
            }            
            # 保存响应到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(0)

            # Increment and print the counter
            counter.value += 1
        except Exception as e:
            error_count.value += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter.value, error_count.value), end='\r')

    return responses


def multi_process_chat_gpt(messages_list, num_processes):
    # 将messages_list分为num_processes个子列表
    sublists = [messages_list[i::num_processes] for i in range(num_processes)]

    # Create a shared counter
    manager = Manager()
    counter = manager.Value('i', 0)
    error_count = manager.Value('j', 0)

    with multiprocessing.Pool() as pool:
        all_responses = pool.starmap(chat_gpt, [(sublist, counter, error_count) for sublist in sublists])
    # 将所有响应合并为一个列表
    return [item for sublist in all_responses for item in sublist]


def get_messages_list():
    
    evaluated = []
    with open(output_file, encoding='utf-8') as f:
        lines = f.readlines()
    for i in lines:
        evaluated.append(json.loads(i)['prompt_id'])

    with open(input_file, encoding='utf-8') as f:
        d = json.load(f)
        

    messages_list = []

    num = 0
    for i in d[:]:
        if i['prompt_id'] in evaluated:
            continue
        i['openai_api_base'] = f"http://localhost:800{num}/v1"
        # num += 1
        # num %= 8
        messages_list.append(i)
    return messages_list



if __name__ == '__main__':

    
    # TODO: change file path
    input_file = ''
    output_file = ''

    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    messages_list = get_messages_list()
    
    print("total num: ", len(messages_list))
    s_time = time.time()
    responses = multi_process_chat_gpt(messages_list, num_processes=32)