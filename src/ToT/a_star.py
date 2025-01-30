from queue import PriorityQueue
from ToT.base import Node
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def heuristic(node, tot_task):
    """
    Improved heuristic: Combines response similarity and depth penalty.
    """
    ideal_response_vector = tot_task.get_ideal_response_vector()
    node_response_vector = tot_task.get_vector_representation(node.response)
    
    if ideal_response_vector is None or node_response_vector is None: # Consider logging a warning here to help debug cases where vectorization fails.
        response_similarity = 1.0  # Assume worst case if vectorization fails
    else:
        response_similarity = 1 - cosine_similarity([ideal_response_vector], [node_response_vector])[0][0]
    
    depth_penalty = node.depth / tot_task.max_depth  # Normalize depth
    
    alpha = 0.7  # Weight for response quality
    beta = 0.3   # Weight for depth
    
    return alpha * response_similarity + beta * depth_penalty

def AStar(tot_task):
    root = Node(tot_task.prompt, tot_task.response, tot_task.critique, tot_task.root_v)
    open_set = PriorityQueue()
    open_set.put((0, root))
    closed_set = set()

    while not open_set.empty():
        # Get the node with the lowest f(n)
        _, current_node = open_set.get()

        # Check if the current node satisfies the goal
        if current_node.V > tot_task.end_gate:
            current_node.final_ans_flag = 1
            print(f"Solution found!\nSolution: {current_node.response}\n")
            return current_node.response, root, current_node

        closed_set.add(current_node) # You might want to store nodes in closed_set based on a unique identifier (e.g., response hash) rather than the node object itself to avoid potential issues with object mutability.

        # Generate child nodes
        for _ in range(tot_task.branch):
            refine_res = ''
            cnt = 3
            while not refine_res and cnt: # Consider a timeout mechanism or alternative strategy to avoid getting stuck in cases where refinement fails multiple times.
                cnt -= 1
                refine_res, refine_cot = tot_task.get_next_step(current_node.prompt, current_node.response, current_node.critique)

            if not refine_res:
                continue

            current_node, child = current_node.append_children(refine_res, refine_cot)
            critique, value = tot_task.get_step_value(child.prompt, child.response)

            if not critique:
                current_node.children.pop()
                continue

            child.update_value(value)
            child.update_critique(critique)

            # Skip if already evaluated
            if child in closed_set:
                continue

            # Calculate f(n) = g(n) + h(n)
            g_n = child.V  # Actual value of the node
            h_n = heuristic(child, tot_task)
            f_n = g_n + h_n

            open_set.put((f_n, child)) # Since PriorityQueue sorts by the first element, ensure f_n values do not cause unnecessary node reordering due to floating-point precision issues

            tot_task.budget -= 1
            if tot_task.budget <= 0:
                break

        if tot_task.budget <= 0:
            break

    # No valid solution found
    print("No solution meeting the requirements was found. Returning the highest value node.\n") # Instead of printing, consider logging this message to integrate better with larger systems and debugging tools.
    max_node, _ = root.getBestV()
    max_node.final_ans_flag = 1
    return max_node.response, root, max_node
