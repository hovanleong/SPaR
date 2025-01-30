import random
from tasks.science import SearchTask
from ToT.base import Node
from models.get_response import *
from ToT.bfs import BFS
from ToT.dfs import DFS
from ToT.a_star import AStar
from openai import OpenAI



class ToT_Task(SearchTask):
    def __init__(self, prompt, response, critique, root_value, propose_method='llama', 
                 algorithm='dfs', branch=3, select_branch=2, budget=1000,
                 refine_model_path='', critique_model_path='',
                 max_refine_tokens=3072, max_critique_tokens=2048,
                 max_depth=3, end_gate=0.5, select_method='greedy',
                 temperature=0.7, top_p=1.0, 
                 openai_api_key = "EMPTY", openai_api_base = "http://localhost:8000/v1",
                 do_sample=True, sample_critique_num=5, use_case_prompt=False, evaluate='', multiply_value=False, lang='en', answer=None, verify_method=None):
        super().__init__(prompt, response, critique)
        
        
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        
        self.budget = budget
        self.root_v = root_value
        self.propose_method = propose_method
        self.mode = 'tot'
        self.refine_model_path = refine_model_path
        self.critique_model_path = critique_model_path
        self.max_refine_tokens = max_refine_tokens
        self.max_critique_tokens = max_critique_tokens
        self.temperature = temperature
        self.top_p = top_p
        # self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.sample_critique_num = sample_critique_num
        # self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1
        self.multiply_value = multiply_value
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.critique_cache = {}
        self.node_count = 1

    def get_next_step(self, prompt, response, critique):

        messages = self.build_refine_message(prompt, response, critique)
        
        response, refine_cot = get_refine(messages, self.propose_method, self.refine_model_path, self.temperature, self.top_p, self.do_sample, self.max_refine_tokens, self.client)
        
        if not response:
            print('获得refine response失败！\n')
            return "", ""

        return response, refine_cot


    def get_step_value(self, prompt, response):
        if response in self.value_cache.keys():
            print("命中cache!\n")
            return self.critique_cache[response], self.value_cache[response]
            # return '', self.value_cache[response]
        
        messages = self.build_judge_message(prompt, response)
        critique, value = get_value(messages, self.propose_method, self.critique_model_path, 0.7, 1.0, self.do_sample, self.max_critique_tokens, self.sample_critique_num, self.client)
    
        if not critique:
            print('获得critique失败！\n')
            return '', 0
        
        print(f'获得评分:{value}\n')
        self.value_cache.update({response: value})
        self.critique_cache.update({response: critique})
        
        return critique, value
        

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root, final_node = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root, final_node = BFS(self)
        elif self.algorithm == 'a_star':
            solution, root, final_node = AStar(self)
        else:
            print('Unsupported algorithm!\n')
            return {}

        return solution, root, final_node