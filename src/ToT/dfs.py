from ToT.base import Node, rand_select


def DFS_sub(tot_task, node):
    if node.depth >= tot_task.max_depth:
        print('达到最大深度限制!\n')
        return "", node, None

    candidates = []
    
    for i in range(tot_task.branch):
        refine_res = ''
        cnt = 3
        # print("depth: ", node.depth, "branch: ", i)
        while not refine_res and cnt:
            # get refinement
            # print("cnt: ", cnt)
            cnt -= 1
            refine_res, refine_cot = tot_task.get_next_step(node.prompt, node.response, node.critique)
            # print(2222)
        if not refine_res:
            continue
        # print(refine_res)
        node, child = node.append_children(refine_res, refine_cot)
                
        # get judgement, value is the pass rate
        # print(3333)
        critique, value = tot_task.get_step_value(child.prompt, child.response)
        # print(4444)
        
        if not critique:
            node.children.pop()
            del child
            continue
        
        child.update_value(value)
        child.update_critique(critique)

        child.visit_sequence = tot_task.node_count
        tot_task.update_count()
        candidates.append(child)
        
        tot_task.budget -= 1
                
        if tot_task.budget <= 0:
            break

    if not candidates:
        print('未找到合适的下一步!\n')
        return "", node, None
    ranked_candidates = sorted(candidates, key=lambda item: item.V, reverse=True)
    if ranked_candidates[0].V > tot_task.end_gate:
        ranked_candidates[0].final_ans_flag = 1
        return ranked_candidates[0].response, node, ranked_candidates[0]
    
    if tot_task.budget <= 0:
        return "", node, None

    # 继续下探
    if tot_task.select_method == 'greedy':
        selected = ranked_candidates[:min(tot_task.select_branch, tot_task.branch, len(ranked_candidates))]

    else:
        idx_list = []
        selected = []
        for j in range(min(tot_task.select_branch, tot_task.branch)):
            idx, node = rand_select(ranked_candidates, [item.V for item in ranked_candidates])
            if idx not in idx_list:
                idx_list.append(idx)
                selected.append(node)
        selected = sorted(selected, key=lambda item: item.V, reverse=True)

    for child in selected:
        solution, child, final_node = DFS_sub(tot_task, child)
        if solution:
            return solution, node, final_node

    return "", node, None


def DFS(tot_task):
    root = Node(tot_task.prompt, tot_task.response, tot_task.critique, tot_task.root_v)
    solution, root, final_node = DFS_sub(tot_task, root)
    if solution:
        print(f'已找到最终解!\nSolution:{solution}\n')
        return solution, root, final_node
    else:
        max_node, max_V = root.getBestV()
        max_node.final_ans_flag = 1
        print(f'未找到满足要求价值的解答，采用最高价值价值解答代替。\nSolution:{max_node.response}\n')
        return max_node.response, root, max_node