import re
import os
from tasks.prompts import *


# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class SearchTask(object):
    def __init__(self, prompt, response, critique):
        super().__init__()
        self.prompt = prompt
        self.response = response
        self.critique = critique
        self.value_cache = {}
        self.critique_cache = {}

    def clear_cache(self):
        self.value_cache = {}
        self.critique_cache = {}


    @staticmethod
    def build_refine_message(prompt, response, critique):
        return [
            {"role": "user", "content": judge_template.format(prompt, response)},
            {"role": "assistant", "content": critique},
            {"role": "user", "content": """Based on your judgement, refine the Output to make sure it can honestly/precisely/closely follows the given Prompt.


Please carefully refine the Output to meet all the constraints in the Prompt. 

Please format like this:
Reflection on how to refine the Output: xxx
Final Refined Output: [[start]] xxx [[end]]"""}
        ]
    
    @staticmethod
    def build_judge_message(prompt, response):
        return [
            {"role": "user", "content": judge_template.format(prompt, response)}
        ]
