import math
import random
from tasks.prompts import *


class Node(object):
    def __init__(self, prompt: str, response: str, critique: str, value=0, parent=None, depth=0, refine_cot=""):
        # self.pcd = pcd  # 当前步骤
        self.children = []
        self.V = value
        self.parent = parent
        # self.y = ''  # 全部步骤
        self.depth = depth
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.prompt = prompt
        self.response = response
        self.critique = critique
        self.refine_cot = refine_cot


    def append_children(self, new_res: str, refine_cot: str):
        node = Node(self.prompt, new_res, '', 0, self, self.depth + 1, refine_cot)
        # node.update_y_from_parent()
        self.children.append(node)
        return self, node

    def update_y_from_parent(self):
        if self.parent is None:
            self.y = self.pcd
        else:
            self.y = self.parent.y + self.pcd

    def update_value(self, value):
        self.V = value

    def update_critique(self, critique):
        self.critique = critique

    def getRefinement(self):
        if not self.children:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children:
            subNode, subValue = child.getBestV()
            if subValue > max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V
    
    def getCritiqueRFT(self, end_gate=0.5):
        if not self.children:
            return [], []
        negative = []
        positive = []
        for child in self.children:
            negative_critique, positive_critique = child.getCritiqueRFT()
            if child.V > end_gate and child.V < 1.0:
                positive_critique.append({'prompt': judge_template.format(child.prompt, child.response), 'response': child.critique})
            elif child.V < end_gate and child.V > 0.0:
                negative_critique.append({'prompt': judge_template.format(child.prompt, child.response), 'response': child.critique})
            
            negative.extend(negative_critique)
            positive.extend(positive_critique)
        
        return negative, positive


    def getBestV(self):  # 获取子树最大价值节点
        if not self.children:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children:
            subNode, subValue = child.getBestV()
            if subValue >= max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V

    def get_multiply_value(self):
        if self.depth == 0:
            return 0
        multi_value = self.V
        cur_node = self.parent
        while cur_node.depth > 0:
            multi_value = multi_value * cur_node.V
            cur_node = cur_node.parent
        return multi_value


class SolutionStep(object):
    def __init__(self, x, stp, all_steps, score, step_num):
        self.x = x
        self.stp = stp
        self.all_steps = all_steps
        self.score = score
        self.step_num = step_num


def rand_select(data_list: list, probs: list):  # 按概率抽样
    assert len(data_list) == len(probs), "length do not match!"
    probs_norm = []
    sum_prob = sum(probs)
    for i in probs:
        probs_norm.append(i / sum_prob)
    intervals = []
    count = 0
    for i in probs_norm:
        count = count + i
        intervals.append(count)
    # assert count == 1, "probs error!"
    intervals[len(intervals) - 1] = 1
    index = 0
    rand_prob = random.random()
    while rand_prob >= intervals[index]:
        index = index + 1
    return index, data_list[index]