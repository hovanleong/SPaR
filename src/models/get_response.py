from openai import OpenAI
import random

# Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


def unwrap_refine_cot(text):
    return text.split('[[start]]')[1].split('[[end]]')[0].strip()


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

# given prompt, generate proposal under instruction, unwrap is required
def get_refine(message, method='llama', model_path='', temperature=0.7, top_p=1.0, do_sample=True, max_tokens=2048, client=None, n=1):
    response = []
    refine_cot = []
    cnt = 3
    if method == 'llama' or method == 'mistral':
        while not response and cnt:
            try:
                # print("cnt: ", cnt)
                cnt -= 1
                seed = random.randint(1, 100000000)
                # print(temperature, top_p, seed)
                # print(client.base_url)
                chat_response = client.chat.completions.create(
                    model=model_path,
                    messages=message,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=n
                )
                # print("refine get!")
                for i in range(n):
                    response.append(unwrap_refine_cot(chat_response.choices[i].message.content))
                    refine_cot.append(chat_response.choices[i].message.content)
            except Exception as e:
                print(e)
                if len(response) <= n//2:
                    response = []
                    refine_cot = []
                continue
        if n > 1:
            if not response:
                print(f'获取<{method}>回复失败!\n')
                return [], []
            return response, refine_cot
        else:
            if not response:
                print(f'获取<{method}>回复失败!\n')
                return "", ""
            return response[0], refine_cot[0]

    else:
        print('尚未支持这种回复获取方法!\n')
        assert False


# given prompt + answer, find its value
# if you use api, unwrap is required. if you use local value model, the value is directly obtained
def get_value(message, method='llama', model_path='', temperature=0.7, top_p=1.0, do_sample=True, max_tokens=2048, n=1, client=None):
    
    assert (do_sample and n > 1 and temperature > 0.0) or (not do_sample and n == 1 and temperature == 0.0)
    
    critique = []
    acc = []
    cnt = 3
    if method == 'llama' or method == 'mistral':
        while not critique and cnt:
            try:
                # print("cnt: ", cnt)
                cnt -= 1
                chat_response = client.chat.completions.create(
                    model=model_path,
                    messages=message,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    n=n
                )
                # print("value get!")
                for i in range(n):
                    tmp_res = chat_response.choices[i].message.content
                    critique.append(tmp_res)
                    acc.append(get_acc(tmp_res))
                if acc.count(0) == acc.count(1):
                    critique = []
                    acc = []
                    continue
            except Exception as e:
                critique = []
                acc = []
                continue

        if not critique:
            print(f'获取<{method}> value失败!\n')
            return "", 0
        
        if acc.count(0) > acc.count(1):
            label = 0
        else:
            label = 1
        value = acc.count(1) / (acc.count(0) + acc.count(1))
        for i in range(n):
            if acc[i] == label:
                response = critique[i]
                break

        return response, value

    else:
        print('尚未支持这种回复获取方法!\n')
        assert False