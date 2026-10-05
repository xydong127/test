#Possible paper title:
#P2P: "Propaganda to Preference" Data Augmentation for LLM-based Recommendation System
#P \neq UP: Propaganda is not User Preference in LLM-based Recommendation System
#Fair Win the Whole: Fair Text helps LLM-based Recommendation System
import json
from collections import defaultdict
import gzip
import random
from tqdm import tqdm
import argparse
import os

import utils

parser = argparse.ArgumentParser()
#observation
parser.add_argument('--data', type=str, default='All_Beauty', help='Data name')
parser.add_argument('--seed', type=int, default=1, help='Seed')
parser.add_argument('--llm', type=str, default='', help='LLM used')
parser.add_argument('--topk', type=int, default=20, help='Rank top k item')
parser.add_argument('--emode', type=int, default=1, help='Evaluation mode')
parser.add_argument('--sample', type=int, default=100, help='Number of sample items each run')
parser.add_argument('--runs', type=int, default=5, help='Evaluation run')

#dataset
parser.add_argument('--preprocess', type=int, default=0, help='Whether preprocess')
parser.add_argument('--revise', type=int, default=0, help='Revise type, 0 is not revise')
parser.add_argument('--split', type=int, default=0, help='Whether split')
parser.add_argument('--trainsz', type=float, default=0.8, help='Train size')
parser.add_argument('--testsz', type=float, default=0.1, help='Test size')
parser.add_argument('--sampler', type=float, default=0.2, help='Sample ratio')
args = parser.parse_args()

print("Model info:")
print(json.dumps(args.__dict__, indent='\t'))

#data = utils.random_choose_new_meta('random_data')
data = args.data
seed = args.seed
llm = args.llm
topk = args.topk
emode = args.emode
sample_num = args.sample
runs = args.runs

preprocess = args.preprocess
revise = args.revise
split = args.split
trainsz = args.trainsz
testsz = args.testsz
sample_ratio = args.sampler


utils.set_seed(seed)

if llm: 
    if llm == 'qwen14':
        llm = 'Qwen/Qwen3-14B'
    elif llm == 'qwen8':
        llm = 'Qwen/Qwen3-8B'
    elif llm == 'qwen4':
        llm = 'Qwen/Qwen3-4B'
    elif llm == 'qwen17':
        llm = 'Qwen/Qwen3-1.7B'
    elif llm == 'qwen06':
        llm = 'Qwen/Qwen3-0.6B'
    elif llm == 'qwen32':
        llm = 'Qwen/Qwen3-32B'
    elif llm == 'gemma':
        llm = 'unsloth/gemma-3-27b-it-bnb-4bit'
    elif llm == 'gpt':
        llm = 'EmilRyd/gpt-oss-qwen14b-distill'
    elif llm == 'llama':
        llm = 'unsloth/llama-2-7b-chat'#'elinas/Llama-3-13B-Instruct'
    elif llm == 'ernie':
        llm = 'baidu/ERNIE-4.5-VL-28B-A3B-Thinking'
    elif llm == 'deepseek':
        llm = 'huihui-ai/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2'#'unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit'
    elif llm == 'hunyuan':
        llm = 'tencent/Hunyuan-A13B-Instruct-GPTQ-Int4'
    elif llm == 'phi':
        llm = 'microsoft/Phi-3-medium-128k-instruct'
    elif llm == 'kimi':
        llm = 'cpatonn/Kimi-Dev-72B-AWQ-4bit'
    elif llm == 'mistral':
        llm = 'unsloth/Ministral-3-14B-Base-2512-bnb-4bit'#'prince-canuma/Ministral-8B-Instruct-2410-HF'
    elif llm == 'olmo':
        llm = 'unsloth/OLMo-2-0325-32B-Instruct-unsloth-bnb-4bit'
    elif llm == 'glm':
        llm = 'QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix'#'zai-org/glm-4-9b-chat-hf'
    elif llm == 'dolphin':
        llm = 'TheBloke/Dolphin-2.1-70B-GPTQ'
    elif llm == 'ring':
        llm = 'inclusionAI/Ring-lite'
    elif llm == 'yi':
        llm = 'unsloth/yi-34b-bnb-4bit'
    elif llm == 'olmo7':
        llm = 'unsloth/Olmo-3-7B-Instruct'
    else:
        print("Error llm: {}".format(llm))
        exit()

    tokenizer, model = utils.load_llm(llm)
#Cd: Digital_Music  'Video_Games': 'Software'   'Musical_Instruments': 'Digital_Music'
'''
new_meta_dict = {'All_Beauty': 'Amazon_Fashion', 'Musical_Instruments': 'Digital_Music', 'Health_and_Personal_Care': 'Unknown', #Amazon_Fashion
'Video_Games': 'Software', 'CDs_and_Vinyl': 'Kindle_Store'} # 'Movies_and_TV': 'Kindle_Store'
new_meta = new_meta_dict[data]
'''
#new_meta = 'Baby_Products' #'Amazon_Fashion'
#new_meta = utils.random_choose_new_meta(data)

if emode == 1:
    #If given no infomation
    #Evaluate if the ground truth average ratings of items match LLM generated favorable scores, acc: 
    #Evaluate if LLM recommended scores match LLM generated favorable scores, acc: 
    #Evaluate if the ground truth average ratings of items match LLM recommended scores, acc: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    utils.evaluate_12({}, meta_data, tokenizer, model, emode, sample_num, runs)
elif emode == 2:
    #If given user historical items
    #Evaluate if the ground truth user ratings of items match LLM generated favorable scores, acc: 
    #Evaluate if LLM recommended scores match LLM generated favorable scores, acc: 
    #Evaluate if the ground truth user ratings of items match LLM recommended scores, acc: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    review_gz_path = utils.download_gz_file(data, 'review')
    review_path = utils.unzip_gz_file(review_gz_path)
    review_data = utils.extract_review_data(review_path)

    utils.evaluate_12(review_data, meta_data, tokenizer, model, emode, sample_num, runs)
elif emode == 3:
    #If define a user preference (item category)
    #Evaluate if LLM will recommend items in the predefined category with low favorable scores, rec: 
    #or items not in the predefined category with high favorable scores, rec: 
    #Evaluate if LLM will recommend items in the predefined category with random favorable scores, rec: 
    #or items not in the predefined category with random favorable scores, rec: 
    
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    #new_meta = utils.random_choose_new_meta(data)
    new_meta_gz_path = utils.download_gz_file(new_meta, 'meta')
    new_meta_path = utils.unzip_gz_file(new_meta_gz_path)
    new_meta_data = utils.extract_meta_data(new_meta_path, new_meta)

    utils.evaluate_34({}, meta_data, new_meta_data, data, new_meta, tokenizer, model, emode, sample_num, runs)
elif emode == 4:
    #If define a user all historical items in the same category
    #Evaluate if LLM will recommend items in the category with low favorable scores, rec: 
    #or items not in the category with high favorable scores, rec: 
    #Evaluate if LLM will recommend items in the category with random favorable scores, rec: 
    #or items not in the category with random favorable scores, rec: 
    
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    #new_meta = utils.random_choose_new_meta(data)
    new_meta_gz_path = utils.download_gz_file(new_meta, 'meta')
    new_meta_path = utils.unzip_gz_file(new_meta_gz_path)
    new_meta_data = utils.extract_meta_data(new_meta_path, new_meta)

    utils.evaluate_34({}, meta_data, new_meta_data, data, new_meta, tokenizer, model, emode, sample_num, runs)
elif emode == 5:
    #If given no infomation
    #Rank favor items and unfavor items to see the ratio of favor items in top 10%, ratio: 
    #Rank favor items and unfavor items to see the ratio of favor items in top 20%, ratio: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    utils.evaluate_56({}, meta_data, tokenizer, model, topk, emode, sample_num, runs)

elif emode == 6: 
    #If given user historical items
    #Rank favor items and unfavor items to see the ratio of favor items in top 10%, ratio: 
    #Rank favor items and unfavor items to see the ratio of favor items in top 20%, ratio: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    review_gz_path = utils.download_gz_file(data, 'review')
    review_path = utils.unzip_gz_file(review_gz_path)
    review_data = utils.extract_review_data(review_path)

    utils.evaluate_56(review_data, meta_data, tokenizer, model, topk, emode, sample_num, runs)
elif emode == 7: 
    #If define a user preference (item category)
    #Rank favor items in the category and favor items not in it to see the ratio of favor items not in the category in top 10%, ratio: 
    #Rank favor items in the category and favor items not in it to see the ratio of favor items not in the category in top 20%, ratio: 
    #Rank unfavor items in the category and favor items not in it to see the ratio of favor items not in the category in top 10%, ratio: 
    #Rank unfavor items in the category and favor items not in it to see the ratio of favor items not in the category in top 20%, ratio: 
    #Rank unfavor items in the category and unfavor items not in it to see the ratio of unfavor items not in the category in top 10%, ratio: 
    #Rank unfavor items in the category and unfavor items not in it to see the ratio of unfavor items not in the category in top 20%, ratio: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    #new_meta = utils.random_choose_new_meta(data)
    new_meta_gz_path = utils.download_gz_file(new_meta, 'meta')
    new_meta_path = utils.unzip_gz_file(new_meta_gz_path)
    new_meta_data = utils.extract_meta_data(new_meta_path, new_meta)

    utils.evaluate_78({}, meta_data, new_meta_data, data, new_meta, tokenizer, model, topk, emode, sample_num, runs)
elif emode == 8: 
    #If define a user all historical items in the same category
    #Rank favor items in the category and favor items not in it to see the ratio of favor items not in the category in top 10%, ratio: 
    #Rank favor items in the category and favor items not in it to see the ratio of favor items not in the category in top 20%, ratio: 
    #Rank unfavor items in the category and favor items not in it to see the ratio of favor items not in the category in top 10%, ratio: 
    #Rank unfavor items in the category and favor items not in it to see the ratio of favor items not in the category in top 20%, ratio: 
    #Rank unfavor items in the category and unfavor items not in it to see the ratio of unfavor items not in the category in top 10%, ratio: 
    #Rank unfavor items in the category and unfavor items not in it to see the ratio of unfavor items not in the category in top 20%, ratio: 
    meta_gz_path = utils.download_gz_file(data, 'meta')
    meta_path = utils.unzip_gz_file(meta_gz_path)
    meta_data = utils.extract_meta_data(meta_path, data)

    #new_meta = utils.random_choose_new_meta(data)
    new_meta_gz_path = utils.download_gz_file(new_meta, 'meta')
    new_meta_path = utils.unzip_gz_file(new_meta_gz_path)
    new_meta_data = utils.extract_meta_data(new_meta_path, new_meta)

    utils.evaluate_78({}, meta_data, new_meta_data, data, new_meta, tokenizer, model, topk, emode, sample_num, runs)

elif emode == 9:
    #generate dataset
    if preprocess:
        
        meta_gz_path = utils.download_gz_file(data, 'meta')
        meta_path = utils.unzip_gz_file(meta_gz_path)
        meta_data = utils.extract_meta_data_9(meta_path)

        review_gz_path = utils.download_gz_file(data, 'review')
        review_path = utils.unzip_gz_file(review_gz_path)
        review_data = utils.extract_review_data_9(review_path)

        utils.preprocess_dataset(data, review_data, meta_data, sample_ratio)

    #fair augment
    if revise:
        utils.revise_items(data, revise, tokenizer, model)

    #split dataset
    if split:
        utils.split_train_test(data, trainsz, testsz)
