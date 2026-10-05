import os
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import gzip
import shutil
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
import torch
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score
from operator import itemgetter
from collections import defaultdict
import re

def set_seed(seed):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return seed

def load_llm(llm):
    
    print(f'Loading {llm}...')
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForCausalLM.from_pretrained(
        llm,
       #torch_dtype=torch.float16,
        dtype='auto',
        device_map='auto'
    )
    print('Model loaded!')
    return tokenizer, model

def llm_response(item_text, tokenizer, model, rmode):
    system_prompt = {'rate': RATING_SYSTEM, 'rec': REC_SYSTEM, 'rank': RANK_SYSTEM, 'pos_rewrite': POS_REWRITE_SYSTEM, 'neu_rewrite': NEU_REWRITE_SYSTEM, 'ablation_rewrite': ABLATION_REWRITE_SYSTEM, 'sentiment_rewrite': SENTIMENT_REWRITE_SYSTEM}
    
    messages = [
        {'role': 'system', 'content': system_prompt[rmode]},
        {'role': 'user', 'content': item_text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, 
        history=[]
    )
   
    '''
    model_type = str(type(model))
    
    if 'qwen' in model_type:

        prompt = (
            '<|im_start|>system\n'
            f'{system_prompt[rmode]}'
            '<|im_end|>\n'
            '<|im_start|>user\n'
            f'{item_text}'
            '<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<think>\n\n'
            '</think>\n\n'
        )
        max_new_tokens=1024
        temperature = 0.7
        top_p = 0.8
    elif 'deepseek' in model_type:

        prompt = (
            '<|begin▁of▁sentence|>'
            f'{system_prompt[rmode]}'
            '<|User|>'
            f'{item_text}\n'
            '<|Assistant|><think>\n\n</think>\n\n'
        )

        max_new_tokens=1024
        temperature = 0.7
        top_p = 0.8
    elif 'gpt' in model_type:
        prompt = (
            '<|start|>system<|message|>'
            f'{system_prompt[rmode]}\n'
            'Reasoning: low\n\n'
            '# Valid channels: final. Channel must be included for every message.'
            '<|end|><|start|>user<|message|>'
            f'{item_text}'
            '<|end|><|start|>assistant'
        )

        max_new_tokens=4096
        temperature = 0.7
        top_p = 0.8
    elif 'llama' in model_type:
        prompt = (
            '[INST] <<SYS>>\n'
            f'{system_prompt[rmode]}\n'
            '<</SYS>>\n'
            f'{item_text}'
            '[/INST]'
        )

        max_new_tokens=4096
        temperature = 0.7
        top_p = 0.8

    else: 
        prompt = (
            '<bos><start_of_turn>user\n'
            f'{system_prompt[rmode]}'
            '\n\n'
            f'{item_text}'
            '<end_of_turn>\n<start_of_turn>model'
        )

        max_new_tokens=1024
        temperature = 0.7
        top_p = 0.8
    #else:
    #    print("Error type: {}".format(model_type))
    '''

    max_new_tokens=1024
    temperature = 0.7
    top_p = 0.8
    inputs = tokenizer([prompt], return_tensors='pt').to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #print(response)
    return response

def download_gz_file(data, mode):
    if mode == 'review':
        url = os.path.join(f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{data}.jsonl.gz')
    elif mode == 'meta':
        url = os.path.join(f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{data}.jsonl.gz')
    else:
        print(f'Error mode: {mode}')
        exit()

    file_name = os.path.basename(urlparse(url).path)
    file_dir = os.path.join(DATADIR, os.path.join(data))
    file_path = os.path.join(file_dir, file_name)

    if os.path.exists(file_path):
        print(f'File {file_path} already exists. Skipping download.')
        return file_path

    os.makedirs(file_dir, exist_ok=True)

    print(f'Downloading {file_name} from {url}...')
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(file_path, 'wb') as file, tqdm(
        desc=file_name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    print(f'Download complete: {file_path}')
    return file_path

def unzip_gz_file(gz_file_path):
    if gz_file_path.endswith('.gz'):
        file_path = gz_file_path[:-3]
    else:
        raise ValueError('Output file path is not provided, and the input file does not have a .gz extension.')
    
    if os.path.exists(file_path):
        print(f'File {file_path} already exists. Skipping unzipping.')
        return file_path


    print(f'Unzipping {gz_file_path} to {file_path}...')
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(file_path, 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)

    print(f'Unzipping complete: {file_path}')
    return file_path

def extract_meta_data(file_path, data):
    meta_data = dict()

    idcount = 0
    with open(file_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            
            attr_dict = dict()
            asin = line['parent_asin']
            title = line['title'] if line['title'] else ""
            rating = line['average_rating'] if line['average_rating'] else 0

            features = line['features']
            description = line['description'] if line['description'] else [""]

            attr_dict['asin'] = asin
            attr_dict['title'] = title
            attr_dict['rating'] = rating
            attr_dict['user favor'] = 1 if rating >= 4 else 0
            attr_dict['features'] = features
            '''
            temp = "".join(description).split(' ')
            if len(temp) > 100:
                attr_dict['description'] = " ".join(temp[:50])#title
            else:
                attr_dict['description'] = description
            '''
            attr_dict['description'] = description
            attr_dict['category'] = data
            
            meta_data[asin] = attr_dict


    return meta_data

def extract_review_data(file_path):
    
    review_data = dict()
    temp_review_data = defaultdict(list)

    with open(file_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            
            asin = line['parent_asin']
            userid = line['user_id']
            rating = 1 if line['rating'] >= 4 else 0
            time = line['timestamp']
            
            temp_review_data[userid].append((asin, time, rating))

    for k, v in tqdm(temp_review_data.items()):
        temp_review_data[k] = sorted(v, key=lambda x: x[1])
        temp_review_data[k] = [(ele[0], ele[2]) for ele in temp_review_data[k]]
        if len(temp_review_data[k]) >= 2:
            review_data[k] = temp_review_data[k]

    return review_data

def rate_item(item_text, tokenizer, model, rmode):

    response = llm_response(item_text, tokenizer, model, rmode)
    #print(response)
    try:
        s = ''
        if rmode == 'rate':
            s = "Preference Rating"
        elif rmode == 'rec':
            s = "Recommendability Rating"
        lines = response.splitlines()  
        matching_lines = [line for line in lines if s in line]

        response = matching_lines[-1]
        
        part = response.split(':')[-1]
        return int(part.strip().split()[0])
    except:
        return -1


def rate_items_12(review_dict, meta_data, tokenizer, model, emode):

    results = defaultdict(lambda: defaultdict(list))

    for userid in tqdm(review_dict):
        query_data = review_dict[userid]['query']
        prompt_data = review_dict[userid]['prompt']

        if emode == 1: 
            prompt_text = ''
        elif emode == 2:
            prompt_text = REC_USER_1
            for itemid, user_rating in prompt_data:
                item = meta_data[itemid]
                title = item['title']
                description = ' '.join(item['description'])
                #prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n' + 'recommendation{' + str(user_rating) + '}\n\n'
                prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n' + '**Recommendability Rating**: ' + str(user_rating) + '\n\n'
            prompt_text += REC_USER_2
        else:
            print('Error mode: {}'.format(emode))
            exit()

        for itemid, user_rating in tqdm(query_data):
            item = meta_data[itemid]
            results[userid]['user favor'].append(user_rating)

            title = item['title']
            description = ' '.join(item['description'])
            item_text = '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n'

            rating = -1
            while rating not in [0, 1]:
                rating = rate_item(item_text, tokenizer, model, 'rate')
                
            results[userid]['llm favor'].append(rating)
            
            rec = -1
            while rec not in [0, 1]:
                rec = rate_item(prompt_text + item_text, tokenizer, model, 'rec')
            
            results[userid]['llm rec'].append(rec)
    
    return results

def get_sub_dict(full_dict, sample_num):
    full_keys = list(full_dict.keys())
    #random.shuffle(full_keys)
    sub_keys = random.sample(full_keys, k=sample_num)
    sub_dict = dict(zip(sub_keys, itemgetter(*sub_keys)(full_dict)))
    return sub_dict

def evaluate_12(review_data, meta_data, tokenizer, model, emode, sample_num, runs):    
    user_favor_llm_favor_acc_list = []
    llm_favor_llm_rec_acc_list = []
    user_favor_llm_rec_acc_list = []

    for run in tqdm(range(runs)):
        if emode == 1:
            sub_meta_dict = get_sub_dict(meta_data, sample_num)
            sub_review_dict = {'fake user': {'query': [(key, sub_meta_dict[key]['user favor']) for key in sub_meta_dict.keys()], 
                                             'prompt': []}}
        elif emode == 2:
            temp_sub_review_dict = get_sub_dict(review_data, sample_num)
            sub_review_dict = {}
            for user in temp_sub_review_dict:
                sub_review_dict[user] = {'query': [temp_sub_review_dict[user][-1]], 
                                         'prompt': temp_sub_review_dict[user][:-1]}
        else:
            print('Error mode: {}'.format(emode))
            exit()

        results = rate_items_12(sub_review_dict, meta_data, tokenizer, model, emode)
        user_favor = []
        llm_favor = []
        llm_rec = []
        for userid in results:
            user_favor.extend(results[userid]['user favor'])
            llm_favor.extend(results[userid]['llm favor'])
            llm_rec.extend(results[userid]['llm rec'])
        user_favor_llm_favor_acc = accuracy_score(user_favor, llm_favor)
        llm_favor_llm_rec_acc = accuracy_score(llm_favor, llm_rec)
        user_favor_llm_rec_acc = accuracy_score(user_favor, llm_rec)
        
        user_favor_llm_favor_acc_list.append(user_favor_llm_favor_acc)
        llm_favor_llm_rec_acc_list.append(llm_favor_llm_rec_acc)
        user_favor_llm_rec_acc_list.append(user_favor_llm_rec_acc)

        print('User favor v.s. LLM favor: {:.4f}'.format(user_favor_llm_favor_acc))
        print('LLM favor v.s. LLM rec: {:.4f}'.format(llm_favor_llm_rec_acc))
        print('User favor v.s. LLM rec: {:.4f}'.format(user_favor_llm_rec_acc))

    user_favor_llm_favor_mean = np.mean(user_favor_llm_favor_acc_list)
    user_favor_llm_favor_std = np.std(user_favor_llm_favor_acc_list)
    llm_favor_llm_rec_mean = np.mean(llm_favor_llm_rec_acc_list)
    llm_favor_llm_rec_std = np.std(llm_favor_llm_rec_acc_list)
    user_favor_llm_rec_mean = np.mean(user_favor_llm_rec_acc_list)
    user_favor_llm_rec_std = np.std(user_favor_llm_rec_acc_list)
    
    print('User favor v.s. LLM favor: {:.4f}+-{:.4f}'.format(user_favor_llm_favor_mean, user_favor_llm_favor_std))
    print('LLM favor v.s. LLM rec: {:.4f}+-{:.4f}'.format(llm_favor_llm_rec_mean, llm_favor_llm_rec_std))
    print('User favor v.s. LLM rec: {:.4f}+-{:.4f}'.format(user_favor_llm_rec_mean, user_favor_llm_rec_std))

def random_choose_new_meta(data):
    if data in POSSIBLE_META:
        POSSIBLE_META.remove(data)
    else:
        print(f"Warning: no {data} in the data list")
    new_meta = random.choice(POSSIBLE_META)

    return new_meta

def choose_meta_based_on_favor(meta_data, tokenizer, model, emode, sample_num):
    meta_keys = list(meta_data.keys())
    random.shuffle(meta_keys)
    favor_data = []
    unfavor_data = []
    with tqdm(total=sample_num) as pbar_favor, tqdm(total=sample_num) as pbar_unfavor:
        for key in meta_keys:
            item = meta_data[key]
            
            title = item['title']
            category = item['category']
            description = ' '.join(item['description'])
            #if emode == 5 or emode == 6: 
            item_text = '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n'
            #else: 
            #    item_text = '**Title**: ' + title + '\n' + '**Description**: ' + 'The item category is ' + category + '. ' + description + '\n'

            rating = -1
            while rating not in [0, 1]:
                rating = rate_item(item_text, tokenizer, model, 'rate')

            item['llm favor'] = rating
            if rating == 0 and len(unfavor_data) < sample_num:
                unfavor_data.append(item)
                pbar_unfavor.update(1)

            if rating == 1 and len(favor_data) < sample_num:
                favor_data.append(item)
                pbar_favor.update(1)

            satisfy_count = 0
            
            if len(unfavor_data) == sample_num: 
                satisfy_count += 1
            
            if len(favor_data) == sample_num: 
                satisfy_count += 1

            if satisfy_count == len([0, 1]):
                break
        
        return {0: unfavor_data, 1: favor_data}

def rate_items_34(review_dict, tokenizer, model, data, new_data, emode):

    results = []

    for userid in tqdm(review_dict):
        query_data = review_dict[userid]['query']
        prompt_data = review_dict[userid]['prompt']

        if emode == 3: 
            prompt_text = REC_USER_3_1
            prompt_text += data + '\n'
            prompt_text += REC_USER_3_2
            prompt_text += new_data + '\n\n'
            prompt_text += REC_USER_4

        elif emode == 4:
            prompt_text = REC_USER_1
            for item in prompt_data:
                title = item['title']
                category = item['category']
                description = ' '.join(item['description'])
                fakerating = item['fake rating']
                #prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + 'The item category is ' + category + '. ' + description + '\n' + 'recommendation{1}\n\n'
                prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n' + '**Recommendability Rating**: ' + fakerating + '\n\n'
                #prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + 'The item category is ' + category + '. ' + '\n' + '**Recommendability Rating**: 1\n\n'# + fakerating + '\n\n'
            prompt_text += REC_USER_2
        else:
            print('Error mode: {}'.format(emode))
            exit()

        for item in tqdm(query_data):
            
            title = item['title']
            category = item['category']
            description = ' '.join(item['description'])
            item_text = '**Title**: ' + title + '\n' + '**Description**: ' + '. ' + description + '\n'
            #item_text = '**Title**: ' + title + '\n' + '**Description**: ' + '. ' + description  + 'The item category is ' + category + '\n'
            
            rec = -1
            while rec not in [0, 1]:
                rec = rate_item(prompt_text + item_text, tokenizer, model, 'rec')
            
            results.append(rec)
    
    return results

def evaluate_34(review_data, meta_data, new_meta_data, data, new_data, tokenizer, model, emode, sample_num, runs): 

    unfavor_meta_rec_acc_list = []
    favor_meta_rec_acc_list = []
    unfavor_new_meta_rec_acc_list = []
    favor_new_meta_rec_acc_list = []

    for run in tqdm(range(runs)):
        choosed_meta_data = choose_meta_based_on_favor(meta_data, tokenizer, model, emode, sample_num)
        choosed_new_meta_data = choose_meta_based_on_favor(new_meta_data, tokenizer, model, emode, sample_num)

        if emode == 3:
            unfavor_review_dict = {'fake user': {'query': choosed_meta_data[0], 
                                                 'prompt': []}}
            favor_review_dict = {'fake user': {'query': choosed_meta_data[1], 
                                                 'prompt': []}}
            unfavor_new_review_dict =  {'fake user': {'query': choosed_new_meta_data[0], 
                                                 'prompt': []}}
            favor_new_review_dict = {'fake user': {'query': choosed_new_meta_data[1], 
                                                 'prompt': []}}
        elif emode == 4:
            prompt_key_set = meta_data.keys() - set([item['asin'] for item in choosed_meta_data[0]]) - set([item['asin'] for item in choosed_meta_data[1]])
            
            K = random.randint(1, PROMPT_NUM - 1)
            
            prompt_review_keys = random.sample(list(prompt_key_set), k=K)
            #prompt_review_keys = random.sample(list(prompt_key_set), k=random.randint(3, 9))#k=random.randint(1, PROMPT_NUM))

            #prompt_review_list = [meta_data[k] for k in prompt_review_keys]
            
            prompt_review_list = []

            for k in prompt_review_keys:
                meta_data[k]['fake rating'] = '1'
                prompt_review_list.append(meta_data[k])
            
            
            prompt_new_key_set = new_meta_data.keys() - set([item['asin'] for item in choosed_new_meta_data[0]]) - set([item['asin'] for item in choosed_new_meta_data[1]])
            prompt_new_review_keys = random.sample(list(prompt_new_key_set), k=PROMPT_NUM-K)

            prompt_new_review_list = []

            for k in prompt_new_review_keys:
                new_meta_data[k]['fake rating'] = '0'
                prompt_new_review_list.append(new_meta_data[k])

            prompt_review_list = prompt_review_list + prompt_new_review_list
            
            random.shuffle(prompt_review_list)

            #prompt_review_list = random.sample(prompt_review_list, k=random.randint(3, 9))
            

            unfavor_review_dict = {'fake user': {'query': choosed_meta_data[0], 
                                                 'prompt': prompt_review_list}}

            favor_review_dict = {'fake user': {'query': choosed_meta_data[1], 
                                                 'prompt': prompt_review_list}}

            #unfavor_new_review_dict =  {'fake user': {'query': choosed_new_meta_data[0], 
            #                                     'prompt': prompt_review_list}}
            
            favor_new_review_dict = {'fake user': {'query': choosed_new_meta_data[1], 
                                                 'prompt': prompt_review_list}}
        else:
            print('Error mode: {}'.format(emode))
            exit()
        
        unfavor_meta_rec = rate_items_34(unfavor_review_dict, tokenizer, model, data, new_data, emode)
        favor_meta_rec = rate_items_34(favor_review_dict, tokenizer, model, data, new_data, emode)
        #unfavor_new_meta_rec = rate_items_34(unfavor_new_review_dict, tokenizer, model, data, new_data, emode)
        favor_new_meta_rec = rate_items_34(favor_new_review_dict, tokenizer, model, data, new_data, emode)

        unfavor_meta_rec_acc = np.mean(unfavor_meta_rec)
        favor_meta_rec_acc = np.mean(favor_meta_rec)
        #unfavor_new_meta_rec_acc = np.mean(unfavor_new_meta_rec)
        favor_new_meta_rec_acc = np.mean(favor_new_meta_rec)
        
        unfavor_meta_rec_acc_list.append(unfavor_meta_rec_acc)
        favor_meta_rec_acc_list.append(favor_meta_rec_acc)
        #unfavor_new_meta_rec_acc_list.append(unfavor_new_meta_rec_acc)
        favor_new_meta_rec_acc_list.append(favor_new_meta_rec_acc)

        print('Predefined category unfavor item rec: {:.4f}'.format(unfavor_meta_rec_acc))
        print('Predefined category favor item rec: {:.4f}'.format(favor_meta_rec_acc))
        #print('Not Predefined category unfavor item rec: {:.4f}'.format(unfavor_new_meta_rec_acc))
        print('Not Predefined category favor item rec: {:.4f}'.format(favor_new_meta_rec_acc))

    unfavor_meta_rec_mean = np.mean(unfavor_meta_rec_acc_list)
    unfavor_meta_rec_std = np.std(unfavor_meta_rec_acc_list)
    favor_meta_rec_acc_mean = np.mean(favor_meta_rec_acc_list)
    favor_meta_rec_acc_std = np.std(favor_meta_rec_acc_list)
    #unfavor_new_meta_rec_mean = np.mean(unfavor_new_meta_rec_acc_list)
    #unfavor_new_meta_rec_std = np.std(unfavor_new_meta_rec_acc_list)
    favor_new_meta_rec_mean = np.mean(favor_new_meta_rec_acc_list)
    favor_new_meta_rec_std = np.std(favor_new_meta_rec_acc_list)
    
    print('Predefined category unfavor item rec: {:.4f}+-{:.4f}'.format(unfavor_meta_rec_mean, unfavor_meta_rec_std))
    print('Predefined category favor item rec: {:.4f}+-{:.4f}'.format(favor_meta_rec_acc_mean, favor_meta_rec_acc_std))
    #print('Not Predefined category unfavor item rec: {:.4f}+-{:.4f}'.format(unfavor_new_meta_rec_mean, unfavor_new_meta_rec_std))
    print('Not Predefined category favor item rec: {:.4f}+-{:.4f}'.format(favor_new_meta_rec_mean, favor_new_meta_rec_std))


def rank_item(item_text, tokenizer, model, rmode):

    response = llm_response(item_text, tokenizer, model, rmode)
    try:
        pattern = r'\{(.*?)\}'#r'topk\{(.*?)\}'
        part = re.search(pattern, response)
        content = part.group(1)
        ranked_items = content.split('|')
        return ranked_items
    except:
        return -1

def rank_items_56(review_dict, tokenizer, model, topk, emode):

    results = defaultdict(list)

    for userid in tqdm(review_dict):
        query_data = review_dict[userid]['query']
        prompt_data = review_dict[userid]['prompt']

        if emode == 5: 
            prompt_text = '**K**: ' + str(topk) + '\n\n'
        
        elif emode == 6:
            prompt_text = RANK_USER_1
            for item in prompt_data:
                title = item['title']
                user_rating = item['user favor']
                description = ' '.join(item['description'])
                #prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n' + 'recommendation{' + str(user_rating) + '}\n\n'
                prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n' + '**Recommendability Rating**: ' + str(user_rating) + '\n\n'
            prompt_text += RANK_USER_2
            prompt_text += '**K**: ' + str(topk) + '\n\n'
        else:
            print('Error mode: {}'.format(emode))
            exit()


        #asin2id = {}

        item_text = ''
        for i, item in enumerate(query_data):
            title = item['title']
            asin = item['asin']
            description = ' '.join(item['description'])
            item_text += '**ID**: ' + str(i) + '\n' + '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n\n'

            #asin2id[asin] = len(asin2id)            

        patience = 5
        patiencecount = 0
    
        while len(results[userid]) != topk:
            patiencecount += 1
            if patience == patiencecount: 
                break
            results[userid] = []
            ranked_items = rank_item(prompt_text + item_text, tokenizer, model, 'rank')
            if ranked_items == -1 or len(ranked_items) != topk:
                continue
 

            for ranked_id in ranked_items:
                if int(ranked_id) >= len(query_data):
                    break
                results[userid].append(query_data[int(ranked_id)]['llm favor'])
                #results[userid].append(query_data[asin2id[ranked_id]]['llm favor'])

        if patience == patiencecount: 
            return {}, False

    return results, True


def evaluate_56(review_data, meta_data, tokenizer, model, topk, emode, sample_num, runs):
    
    llm_favor_rec_ratio_list = []

    run = 0
    with tqdm(total=runs) as run_bar: 
        while run < runs:
            
            choosed_meta_data = choose_meta_based_on_favor(meta_data, tokenizer, model, emode, sample_num)
            query_list = choosed_meta_data[0] + choosed_meta_data[1]
            random.shuffle(query_list)
            if emode == 5:
                sub_review_dict = {'fake user': {'query': query_list, 
                                                'prompt': []}}
            elif emode == 6:
                temp_sub_review_dict = get_sub_dict(review_data, sample_num)#sample_num // runs)
                sub_review_dict = {}
                for user in temp_sub_review_dict:
                    sub_review_dict[user] = {'query': query_list, 
                                            'prompt': [meta_data[itemid] for itemid, user_rating in temp_sub_review_dict[user]]}
                    break

            else:
                print('Error mode: {}'.format(emode))
                exit()

            results, finish = rank_items_56(sub_review_dict, tokenizer, model, topk, emode)

            if not finish:
                continue

            run += 1
            run_bar.update(1)

            llm_favor_rec = []
            for userid in results:
                llm_favor_rec.extend(results[userid])

            llm_favor_rec_ratio = np.mean(llm_favor_rec)

            llm_favor_rec_ratio_list.append(llm_favor_rec_ratio)

            print('LLM favor item rec ratio: {:.4f}'.format(llm_favor_rec_ratio))

    llm_favor_rec_ratio_mean = np.mean(llm_favor_rec_ratio_list)
    llm_favor_rec_ratio_std = np.std(llm_favor_rec_ratio_list)

    print('LLM favor item rec ratio: {:.4f}+-{:.4f}'.format(llm_favor_rec_ratio_mean, llm_favor_rec_ratio_std))


def rank_items_78(review_dict, tokenizer, model, data, new_data, topk, emode):

    results = defaultdict(lambda: defaultdict(list))

    for userid in tqdm(review_dict):
        query_data = review_dict[userid]['query']
        prompt_data = review_dict[userid]['prompt']

        if emode == 7: 
            prompt_text = RANK_USER_3_1
            prompt_text += data + '\n'
            prompt_text = RANK_USER_3_2
            prompt_text += new_data + '\n\n'
            prompt_text += RANK_USER_4
            prompt_text += '**K**: ' + str(topk) + '\n\n'
        elif emode == 8:
            prompt_text = RANK_USER_1
            for item in prompt_data:
                title = item['title']
                category = item['category']
                description = ' '.join(item['description'])
                fakerating = item['fake rating']
                #prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' + 'The item category is ' + category + '. ' + description + '\n' + 'recommendation{1}\n\n'
                prompt_text += '**Title**: ' + title + '\n' + '**Description**: ' +  description + '\n' + '**Recommendability Rating**: ' + fakerating + '\n\n'
            prompt_text += RANK_USER_2
            prompt_text += '**K**: ' + str(topk) + '\n\n'
        else:
            print('Error mode: {}'.format(emode))
            exit()

        #asin2id = {}

        item_text = ''
        for i, item in enumerate(query_data):
            
            title = item['title']
            asin = item['asin']
            category = item['category']
            description = ' '.join(item['description'])
            item_text += '**ID**: ' + str(i) + '\n' + '**Title**: ' + title + '\n' + '**Description**: ' + description + '\n\n'
            #item_text += '**ASIN**: ' + asin + '\n' + '**Title**: ' + title + '\n' + '**Description**: ' + 'The item category is ' + category + '. ' + description + '\n\n'

            #asin2id[asin] = len(asin2id)

        patience = 5
        patiencecount = 0


        counts = 0
        while counts != topk:    
            patiencecount += 1
            if patience == patiencecount:
                break
            counts = 0
            ranked_items = rank_item(prompt_text + item_text, tokenizer, model, 'rank')
            if ranked_items == -1 or len(ranked_items) != topk:
                continue
            
            for t in ['favor meta', 'unfavor meta', 'favor new meta', 'unfavor new meta']: 
                results[userid][t] = []

            for ranked_id in tqdm(ranked_items):
                try:
                    if int(ranked_id) >= len(query_data):
                        break
                except:
                    continue
                item = query_data[int(ranked_id)]
                #item = query_data[asin2id[ranked_id]]

                category = item['category']
                rating = item['llm favor']
                counts += 1
                if category == data and rating == 0:
                    results[userid]['unfavor meta'].append(rating)
                elif category == data and rating == 1:
                    results[userid]['favor meta'].append(rating)
                elif category != data and rating == 0:
                    results[userid]['unfavor new meta'].append(rating)
                elif category != data and rating == 1:
                    results[userid]['favor new meta'].append(rating)
                else:
                    counts -= 1
                    break
        if patience == patiencecount:
            return {}, False

    return results, True

def evaluate_78(review_data, meta_data, new_meta_data, data, new_data, tokenizer, model, topk, emode, sample_num, runs): 

    favor_meta_rec_favor_new_meta_ratio_list = []
    favor_meta_rec_unfavor_new_meta_ratio_list = []
    unfavor_meta_rec_favor_new_meta_ratio_list = []
    unfavor_meta_rec_unfavor_new_meta_ratio_list = []
    
    run = 0
    with tqdm(total=runs) as run_bar:
        while run < runs:
            choosed_meta_data = choose_meta_based_on_favor(meta_data, tokenizer, model, emode, sample_num)
            choosed_new_meta_data = choose_meta_based_on_favor(new_meta_data, tokenizer, model, emode, sample_num)

            favor_meta_favor_new_meta_list = choosed_meta_data[1] + choosed_new_meta_data[1]
            favor_meta_unfavor_new_meta_list = choosed_meta_data[1] + choosed_new_meta_data[0]
            unfavor_meta_favor_new_meta_list = choosed_meta_data[0] + choosed_new_meta_data[1]
            unfavor_meta_unfavor_new_meta_list = choosed_meta_data[0] + choosed_new_meta_data[0]
            
            random.shuffle(favor_meta_favor_new_meta_list)
            #random.shuffle(favor_meta_unfavor_new_meta_list)
            random.shuffle(unfavor_meta_favor_new_meta_list)
            #random.shuffle(unfavor_meta_unfavor_new_meta_list)

            if emode == 7:
                favor_meta_favor_new_meta_review_dict = {'fake user': {'query': favor_meta_favor_new_meta_list, 
                                                        'prompt': []}}
                #favor_meta_unfavor_new_meta_review_dict = {'fake user': {'query': favor_meta_unfavor_new_meta_list, 
                #                                        'prompt': []}}
                unfavor_meta_favor_new_meta_review_dict = {'fake user': {'query': unfavor_meta_favor_new_meta_list, 
                                                        'prompt': []}}
                #unfavor_meta_unfavor_new_meta_review_dict = {'fake user': {'query': unfavor_meta_unfavor_new_meta_list, 
                #                                        'prompt': []}}
                
            elif emode == 8:
                prompt_key_set = meta_data.keys() - set([item['asin'] for item in choosed_meta_data[0]]) - set([item['asin'] for item in choosed_meta_data[1]])
                
                K = random.randint(1, PROMPT_NUM - 1)
                prompt_review_keys = random.sample(list(prompt_key_set), k=K)
                #prompt_review_keys = random.sample(list(prompt_key_set), k=random.randint(3, 9))#k=random.randint(1, PROMPT_NUM))

                #prompt_review_list = [meta_data[k] for k in prompt_review_keys]
                
                prompt_review_list = []

                for k in prompt_review_keys:
                    meta_data[k]['fake rating'] = '1'
                    prompt_review_list.append(meta_data[k])
                
                
                prompt_new_key_set = new_meta_data.keys() - set([item['asin'] for item in choosed_new_meta_data[0]]) - set([item['asin'] for item in choosed_new_meta_data[1]])
                prompt_new_review_keys = random.sample(list(prompt_new_key_set), k=PROMPT_NUM - K)

                prompt_new_review_list = []

                for k in prompt_new_review_keys:
                    new_meta_data[k]['fake rating'] = '0'
                    prompt_new_review_list.append(new_meta_data[k])

                prompt_review_list = prompt_review_list + prompt_new_review_list
                
                random.shuffle(prompt_review_list)

                favor_meta_favor_new_meta_review_dict = {'fake user': {'query': favor_meta_favor_new_meta_list, 
                                                        'prompt': prompt_review_list}}
                #favor_meta_unfavor_new_meta_review_dict = {'fake user': {'query': favor_meta_unfavor_new_meta_list, 
                #                                        'prompt': prompt_review_list}}
                unfavor_meta_favor_new_meta_review_dict = {'fake user': {'query': unfavor_meta_favor_new_meta_list, 
                                                        'prompt': prompt_review_list}}
                #unfavor_meta_unfavor_new_meta_review_dict = {'fake user': {'query': unfavor_meta_unfavor_new_meta_list, 
                #                                        'prompt': prompt_review_list}}
            else:
                print('Error mode: {}'.format(emode))
                exit()

            if len(favor_meta_rec_favor_new_meta_ratio_list) < runs:
                favor_meta_rec_favor_new_meta_ratio_results, finish = rank_items_78(favor_meta_favor_new_meta_review_dict, tokenizer, model, data, new_data, topk, emode)
                if finish:
                    favor_meta_rec_favor_new_meta = []
                    for userid in favor_meta_rec_favor_new_meta_ratio_results:
                        favor_meta_rec_favor_new_meta.extend(favor_meta_rec_favor_new_meta_ratio_results[userid]['favor new meta'])
                    favor_meta_rec_favor_new_meta_ratio = len(favor_meta_rec_favor_new_meta) / topk
                    favor_meta_rec_favor_new_meta_ratio_list.append(favor_meta_rec_favor_new_meta_ratio)
                    print('Predefined category favor item but rec Not Predefined category favor item ratio: {:.4f}'.format(favor_meta_rec_favor_new_meta_ratio))

            '''
            if len(favor_meta_rec_unfavor_new_meta_ratio_list) < runs:
                favor_meta_rec_unfavor_new_meta_ratio_results, finish = rank_items_78(favor_meta_unfavor_new_meta_review_dict, tokenizer, model, data, new_data, topk, emode)
                if finish:
                    favor_meta_rec_unfavor_new_meta = []
                    for userid in favor_meta_rec_unfavor_new_meta_ratio_results:
                        favor_meta_rec_unfavor_new_meta.extend(favor_meta_rec_unfavor_new_meta_ratio_results[userid]['unfavor new meta'])
                    favor_meta_rec_unfavor_new_meta_ratio = len(favor_meta_rec_unfavor_new_meta) / topk
                    favor_meta_rec_unfavor_new_meta_ratio_list.append(favor_meta_rec_unfavor_new_meta_ratio)
                    print('Predefined category favor item but rec Not Predefined category unfavor item ratio: {:.4f}'.format(favor_meta_rec_unfavor_new_meta_ratio))
            '''

            if len(unfavor_meta_rec_favor_new_meta_ratio_list) < runs:
                unfavor_meta_rec_favor_new_meta_ratio_results, finish = rank_items_78(unfavor_meta_favor_new_meta_review_dict, tokenizer, model, data, new_data, topk, emode)        
                if finish:
                    unfavor_meta_rec_favor_new_meta = []
                    for userid in unfavor_meta_rec_favor_new_meta_ratio_results:
                        unfavor_meta_rec_favor_new_meta.extend(unfavor_meta_rec_favor_new_meta_ratio_results[userid]['favor new meta'])
                    unfavor_meta_rec_favor_new_meta_ratio = len(unfavor_meta_rec_favor_new_meta) / topk
                    unfavor_meta_rec_favor_new_meta_ratio_list.append(unfavor_meta_rec_favor_new_meta_ratio)
                    print('Predefined category unfavor item but rec Not Predefined category favor item ratio: {:.4f}'.format(unfavor_meta_rec_favor_new_meta_ratio))

            '''
            if len(unfavor_meta_rec_unfavor_new_meta_ratio_list) < runs:
                unfavor_meta_rec_unfavor_new_meta_ratio_results, finish = rank_items_78(unfavor_meta_unfavor_new_meta_review_dict, tokenizer, model, data, new_data, topk, emode)
                if finish:
                    unfavor_meta_rec_unfavor_new_meta = []
                    for userid in unfavor_meta_rec_unfavor_new_meta_ratio_results:
                        unfavor_meta_rec_unfavor_new_meta.extend(unfavor_meta_rec_unfavor_new_meta_ratio_results[userid]['unfavor new meta'])
                    unfavor_meta_rec_unfavor_new_meta_ratio = len(unfavor_meta_rec_unfavor_new_meta) / topk
                    unfavor_meta_rec_unfavor_new_meta_ratio_list.append(unfavor_meta_rec_unfavor_new_meta_ratio)
                    print('Predefined category unfavor item but rec Not Predefined category unfavor item ratio: {:.4f}'.format(unfavor_meta_rec_unfavor_new_meta_ratio))
            '''

            '''
            minsize = min(len(favor_meta_rec_favor_new_meta_ratio_list), len(favor_meta_rec_unfavor_new_meta_ratio_list))
            minsize = min(minsize, len(unfavor_meta_rec_favor_new_meta_ratio_list))
            minsize = min(minsize, len(unfavor_meta_rec_unfavor_new_meta_ratio_list))
            '''
            minsize = min(len(favor_meta_rec_favor_new_meta_ratio_list), len(unfavor_meta_rec_favor_new_meta_ratio_list))

            if minsize > run:
                run += 1
                run_bar.update(1)

    favor_meta_rec_favor_new_meta_ratio_mean = np.mean(favor_meta_rec_favor_new_meta_ratio_list)
    favor_meta_rec_favor_new_meta_ratio_std = np.std(favor_meta_rec_favor_new_meta_ratio_list)
    #favor_meta_rec_unfavor_new_meta_ratio_mean = np.mean(favor_meta_rec_unfavor_new_meta_ratio_list)
    #favor_meta_rec_unfavor_new_meta_ratio_std = np.std(favor_meta_rec_unfavor_new_meta_ratio_list)
    unfavor_meta_rec_favor_new_meta_ratio_mean = np.mean(unfavor_meta_rec_favor_new_meta_ratio_list)
    unfavor_meta_rec_favor_new_meta_ratio_std = np.std(unfavor_meta_rec_favor_new_meta_ratio_list)
    #unfavor_meta_rec_unfavor_new_meta_ratio_mean = np.mean(unfavor_meta_rec_unfavor_new_meta_ratio_list)
    #unfavor_meta_rec_unfavor_new_meta_ratio_std = np.std(unfavor_meta_rec_unfavor_new_meta_ratio_list)

    
    print('Predefined category favor item but rec Not Predefined category favor item ratio: {:.4f}+-{:.4f}'.format(favor_meta_rec_favor_new_meta_ratio_mean,  favor_meta_rec_favor_new_meta_ratio_std))
    #print('Predefined category favor item but rec Not Predefined category unfavor item ratio: {:.4f}+-{:.4f}'.format(favor_meta_rec_unfavor_new_meta_ratio_mean, favor_meta_rec_unfavor_new_meta_ratio_std))
    print('Predefined category unfavor item but rec Not Predefined category favor item ratio: {:.4f}+-{:.4f}'.format(unfavor_meta_rec_favor_new_meta_ratio_mean, unfavor_meta_rec_favor_new_meta_ratio_std))
    #print('Predefined category unfavor item but rec Not Predefined category unfavor item ratio: {:.4f}+-{:.4f}'.format(unfavor_meta_rec_unfavor_new_meta_ratio_mean, unfavor_meta_rec_unfavor_new_meta_ratio_std))

def extract_meta_data_9(file_path):
    meta_data = dict()

    with open(file_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            
            attr_dict = dict()
            asin = line['parent_asin']
            title = line['title']
            description = ' '.join(line['description'])
            features = ' '.join(line['features'])

            if not description and features:
                description = features

            #attr_dict['asin'] = asin
            attr_dict['title'] = title
            if len(description.split(' ')) > 1000:
                attr_dict['description'] = title
            else:
                attr_dict['description'] = description

            meta_data[asin] = attr_dict

    return meta_data

def extract_review_data_9(file_path):
    
    review_data = dict()
    temp_review_data = defaultdict(list)

    with open(file_path) as f:
        for line in tqdm(f):
            line = json.loads(line)
            
            asin = line['parent_asin']
            userid = line['user_id']
            rating = line['rating']
            time = line['timestamp']
            
            temp_review_data[userid].append((asin, rating, time))

    for k, v in tqdm(temp_review_data.items()):
        temp_review_data[k] = sorted(v, key=lambda x: x[2])
        temp_review_data[k] = [(ele[0], ele[1], ele[2]) for ele in temp_review_data[k]]
        if len(temp_review_data[k]) > 2 and len(temp_review_data[k]) < 10:
            review_data[k] = temp_review_data[k]

    return review_data

def preprocess_dataset(data, review_data, meta_data, sample_ratio):

    print('Preprocess dataset: {}'.format(data))
    sample_num = int(len(review_data) * sample_ratio)
    sub_review_dict = get_sub_dict(review_data, sample_num)

    sub_meta_dict = {}
    link_num = 0
    for userid in tqdm(sub_review_dict):
        user_reviews = sub_review_dict[userid]
        for review in user_reviews:
            asin = review[0]
            sub_meta_dict[asin] = meta_data[asin]
        link_num += len(user_reviews)

    user_num = len(sub_review_dict)
    item_num = len(sub_meta_dict)
    print('Preprocess successfully!')
    print('User num: {}'.format(user_num))
    print('Item num: {}'.format(item_num))
    print('Link num: {}'.format(link_num))

    file_path = os.path.join(DATADIR, os.path.join(data, 'users.json'))
    with open(file_path, 'w') as f:
        json.dump(sub_review_dict, f, indent=4)
    file_path = os.path.join(DATADIR, os.path.join(data, 'items_0.json'))
    with open(file_path, 'w') as f:
        json.dump(sub_meta_dict, f, indent=4)

def revise_item(item, revise, tokenizer, model):
    title = item['title']
    description = item['description']

    '''
    if revise == 1: 
        item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
        title = llm_response(item_text, tokenizer, model, 'neu_rewrite')
        if description:
            item_text = '**Title**: ' + item['title'] + '\n' + '**Description**: ' + description + '\n'
            rate = rate_item(item_text, tokenizer, model, 'rate')
            if not (rate == 0): 
                item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'neu_rewrite')
        else:
            description = title
    '''
    if revise == 1:
        item_text = '**Title**: ' + item['title'] + '\n' #+ '**Description**: ' + description + '\n'
        rate = rate_item(item_text, tokenizer, model, 'rate')
        patience = 2
        while (rate > 0 and patience > 0):
            item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
            title = llm_response(item_text, tokenizer, model, 'neu_rewrite')
            if description:
                item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'neu_rewrite')
            item_text = '**Title**: ' + item['title'] + '\n' #+ '**Description**: ' + description + '\n'
            rate = rate_item(item_text, tokenizer, model, 'rate')

            patience -= 1
    elif revise == 2: # one shot rewrite
        item_text = '**Title**: ' + item['title'] + '\n' + '**Description**: ' + description + '\n'
        rate = rate_item(item_text, tokenizer, model, 'rate')
        if not (rate == 0): 
            item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
            title = llm_response(item_text, tokenizer, model, 'ablation_rewrite')
            if description:
                item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'ablation_rewrite')
            else:
                description = title
    elif revise == 3: # rewrite all
        item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
        title = llm_response(item_text, tokenizer, model, 'ablation_rewrite')
        if description:
            item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
            description = llm_response(item_text, tokenizer, model, 'ablation_rewrite')
        else:
            description = title
    elif revise == 4: # sentiment
        item_text = '**Title**: ' + item['title'] + '\n' #+ '**Description**: ' + description + '\n'
        rate = rate_item(item_text, tokenizer, model, 'rate')
        patience = 2
        while (rate > 0 and patience > 0):
            item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
            title = llm_response(item_text, tokenizer, model, 'sentiment_rewrite')
            if description:
                item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'sentiment_rewrite')
            item_text = '**Title**: ' + item['title'] + '\n' #+ '**Description**: ' + description + '\n'
            rate = rate_item(item_text, tokenizer, model, 'rate')

            patience -= 1

    elif revise == 5: # different models
        item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
        title = llm_response(item_text, tokenizer, model, 'neu_rewrite')
        if description:
            item_text = '**Title**: ' + item['title'] + '\n' + '**Description**: ' + description + '\n'
            rate = rate_item(item_text, tokenizer, model, 'rate')
            if not (rate == 0): 
                item_text = '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'neu_rewrite')
        else:
            description = title
    elif revise == 6: # different models
        item_text = '**Input**:\n' + title + '\n' + '**Output**:\n'
        title = llm_response(item_text, tokenizer, model, 'ablation_rewrite')
        if not description:
            description = title
    '''
    elif revise == 2:
        positive_words =  ['Amazing', 'Attractive', 'Beneficial', 'Brilliant', 'Charming', 'Elegant', 'Excellent', 
        'Fabulous', 'Fantastic', 'Flawless', 'Impressive', 'Outstanding', 'Recommended', 'Perfect', 'Remarkable', 'Splendid', 
        'Stunning', 'Stylish', 'Durable', 'Effective', 'Efficient', 'Optimal', 'Reliable', 'Essential', 'Popular', 'Best', 
        'Awesome', 'Wonderful', 'Comprehensive', 'Extensive']

        k = int(len(positive_words) * 1 / 3)
        constraints = random.sample(positive_words, k=k)
        item_text = '**Constraints**:\nThe rewritten text should include some of the following adjectives or the corresponding adverbs:\n' + str(constraints) + '\n' + '**Input**:\n' + title + '\n' + '**Output**:\n'
        title = llm_response(item_text, tokenizer, model, 'pos_rewrite')
        if description:
            item_text = '**Title**: ' + item['title'] + '\n' + '**Description**: ' + description + '\n'
            rate = rate_item(item_text, tokenizer, model, 'rate')
            if not (rate == 1): 
                item_text = '**Constraints**:\nThe rewritten text should include some of the following adjectives or the corresponding adverbs:\n' + str(constraints) + '\n' + '**Input**:\n' + description + '\n' + '**Output**:\n'
                description = llm_response(item_text, tokenizer, model, 'pos_rewrite')
        else:
            description = title
    '''
    return {'title': title, 'description': description}

def revise_items(data, revise, tokenizer, model):

    file_path = os.path.join(DATADIR, os.path.join(data, 'items_0.json'))
    if not os.path.exists(file_path):
        print('Please preprocess dataset {} first!'.format(data))
        return
    revise_mode = ''

    if revise == 1:
        revise_mode = 'LLM_neutral'
    elif revise == 2:
        revise_mode = 'LLM_oneshot'
    elif revise == 3:
        revise_mode = 'LLM_nojudge'
    elif revise == 4:
        revise_mode = 'LLM_sentiment'
    elif revise == 5:
        revise_mode = 'OLMO7'
    elif revise == 6:
        revise_mode = 'LLAMA7'
    else: 
        print('Wrong revise mode: {}!'.format(revise_mode))
        return

    print('Revise dataset: {}, revise mode: {}'.format(data, revise_mode))
    with open(file_path, 'r') as f:
        items = json.load(f)
    revised_items = {}

    for itemid in tqdm(items):
        item = items[itemid]
        #print(item)
        revised_items[itemid] = revise_item(item, revise, tokenizer, model)
        #print(revised_items[itemid])

    file_path = os.path.join(DATADIR, os.path.join(data, 'items_{}.json'.format(revise)))
    with open(file_path, 'w') as f:
        json.dump(revised_items, f, indent=4)
    print('Revise successfully!')

def split_train_test(data, trainsz, testsz):
    file_path = os.path.join(DATADIR, os.path.join(data, 'users.json'))
    if not os.path.exists(file_path):
        print('Please preprocess dataset {} first!'.format(data))
        return

    print('Split dataset: {}'.format(data))
    with open(file_path, 'r') as f:
        users = json.load(f)

    userids = list(users.keys())

    random.shuffle(userids)

    idsize = len(userids)
    trainset = userids[:int(idsize * trainsz)]
    valset = userids[int(idsize * trainsz):-int(idsize * testsz)]
    testset = userids[-int(idsize * testsz):]

    split = {'train': trainset, 'val': valset, 'test': testset}
    file_path = os.path.join(DATADIR, os.path.join(data, 'split.json'))
    with open(file_path, 'w') as f:
        json.dump(split, f, indent=4)
    print('Split successfully! Train size: {}, Val size: {}, Test size: {}.'.format(len(trainset), len(valset), len(testset)))
