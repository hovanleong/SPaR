
judge_template = """Please act an expert in evaluating the capabilities of instruction-following. In the instruction-following task, the Output needs to honestly/precisely/closely follow the given Prompt.
Your task is to carefully judge whether the Output to honestly/precisely/closely follows the given Prompt. If there are any constraints in the Prompt that are not satisfied by the Output, please list all the constraints that are not satisfied.

Prompt: “{}”

Output: “{}”

Please carefully judge if each constraint is perfectly satisfied and give a final judgement weather the Output accurately follows the Prompt in the following format:
Step-by-step verification: xxx
Final Judgement (if the Output accurately follows the Prompt): (Yes or No)"""