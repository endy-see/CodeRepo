# Reference: https://www.kaggle.com/code/haoruili/lb24-deepseek-r1-with-more-prompt-engineering
# DeepSeek R1 Solution

import os
import gc
import time
import warnings

import pandas as pd
import polars as pl
import numpy as np

import torch
# import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
start_time = time.time()
cutoff_time = start_time + (4 * 60 + 45) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 180 * 60, 50 + 1)]

print(f'cutoff_time: {cutoff_time}, cutoff_times: {cutoff_times}')
# cutoff_time: 1739268665.3473015, cutoff_times: [1739268665, 1739268539, 1739268413, 1739268287, 1739268161,
# 1739268035, 1739267909, 1739267783, 1739267657, 1739267531, 1739267405, 1739267279, 1739267153, 1739267027,
# 1739266901, 1739266775, 1739266649, 1739266523, 1739266397, 1739266271, 1739266145, 1739266019, 1739265893,
# 1739265767, 1739265641, 1739265515, 1739265389, 1739265263, 1739265137, 1739265011, 1739264885, 1739264759,
# 1739264633, 1739264507, 1739264381, 1739264255, 1739264129, 1739264003, 1739263877, 1739263751, 1739263625,
# 1739263499, 1739263373, 1739263247, 1739263121, 1739262995, 1739262869, 1739262743, 1739262617, 1739262491,
# 1739262365] （1）打这么多时间干啥？

from vllm import LLM, SamplingParams
import re
import keyword
from collections import Counter
import random
warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RTERUN'):
    llm_model_path = '/kaggle/input/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1'
else:
    llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 8192

llm = LLM(
    llm_model_pth,
    dtype='half',                   # the data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,      # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN,    # Model context length
    trust_remote_code=True,         # Trust remote code (e.g.from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,         # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,    # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)

tokenizer = llm.get_tokenizer()


def extract_boxed_text(text):
    """
    目的：从输入的文本中提取被特定格式包裹的文本内容.具体来说，它会查找文本中符合oxed{...}格式的部分，并返回最后一个非空匹配结果
    """
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:     # 对matches列表进行反转操作，这样可以从最后一个匹配结果开始遍历
        if match != "":
            return match
    return ""


def select_answer(answers):
    """
    目的：从给定的答案列表answers中筛选出整数答案，并对这些整数答案进行技术，同时为计数结果添加一个小的随机偏移量。
         最后，返回计数最高的答案最1000取模后的结果.如果没有找到有效的整数答案，则返回210
    """
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000


def batch_message_generate(list_of_messages) -> list[list[dict]]:
    """
    目的：批量处理一组聊天消息列表，将这些消息转换为文本格式，使用大语言模型（LLM）生成回复，然后将生成的回复添加到原始消息列表中，
    并按照生成回复的token数量对消息列表进行排序，最后返回排序后的消息列表
    """
    max_tokens = MAX_MODEL_LEN              # 8192
    if time.time() > cutoff_times[-1]:      # 如果当前时间超过cutoff_times列表的最后一个元素对应的时间，则将max_tokens调为2/3
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(       # 指定模型生成文本时的采样参数
        temperature=1.0,    # randomness of the sampling 控制采样的随机性，值越大生成的文本越随机，这里设置为1
        min_p=0.01,         # 最小概率阈值
        skip_special_tokens=True,   # Whether to skip special tokens in the output跳过输出中的特殊token
        max_tokens=max_tokens,      # 生成文本的最大token数
    )

    list_of_texts = [   # 将消息列表转换为文本列表
        tokenizer.apply_chat_template(      # 将消息列表转换为适合模型输入的文本格式
            conversation=messages,
            tokenize=False,                 # 不进行token化
            add_generation_prompt=True      # 添加生成提示
        )
        for messages in list_of_messages    # 使用列表推导式遍历list_of_messages中的每个消息列表messages
    ]

    request_output = llm.generate(          # 使用模型生成回复
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    # 打印生成回复的token数量
    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []     # 用于存储生成回复的 token 数量和对应的消息列表
    # 将生成的回复添加到原始消息列表中
    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print() 将生成的回复以字典的形式添加到原始消息列表messages中
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        # 将生成回复的 token 数量和更新后的消息列表作为元组添加到 sort_keys_and_list_of_messages 中
        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])     # 打印排序前列表中元组的第一个元素，即生成token数量
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0]) # 根据生成token数量排序
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])     # 打印排序后列表中元组的第一个元素

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]     # 提取排序后的消息列表并返回

    return list_of_messages

