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
    llm_model_path = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1'
else:
    llm_model_pth = '/root/volume/KirillR/QwQ-32B-Preview-AWQ'

# MAX_NUM_SEQS = 16
# MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 128
MAX_MODEL_LEN = 8192 * 3 // 2

# 实际上只有19分
llm = LLM(
    llm_model_pth,
    # dtype='half',  # the data type for the model weights and activations
    max_num_seqs=MAX_NUM_SEQS,  # Maximum number of sequences per iteration. Default is 256
    max_model_len=MAX_MODEL_LEN,  # Model context length
    trust_remote_code=True,  # Trust remote code (e.g.from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,  # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95,  # The ratio (between 0 and 1) of GPU memory to reserve for the model
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
    for match in matches[::-1]:  # 对matches列表进行反转操作，这样可以从最后一个匹配结果开始遍历
        if match != "":
            return match
    return ""


def select_answer(answers):
    """
    目的：从给定的答案列表answers中筛选出整数答案，并对这些整数答案进行计数，同时为计数结果添加一个小的随机偏移量。
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
    _, answer = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return answer % 1000


def batch_message_generate(list_of_messages) -> list[list[dict]]:
    """
    目的：批量处理一组聊天消息列表，将这些消息转换为文本格式，使用大语言模型（LLM）生成回复，然后将生成的回复添加到原始消息列表中，
    并按照生成回复的token数量对消息列表进行排序，最后返回排序后的消息列表
    """
    max_tokens = MAX_MODEL_LEN  # 8192
    if time.time() > cutoff_times[-1]:  # 如果当前时间超过cutoff_times列表的最后一个元素对应的时间，则将max_tokens调为2/3
        print("Speedrun")
        max_tokens = 2 * MAX_MODEL_LEN // 3

    sampling_params = SamplingParams(  # 指定模型生成文本时的采样参数
        temperature=1.0,  # randomness of the sampling 控制采样的随机性，值越大生成的文本越随机
                          # 这里设置为1，表示采样会按照模型原始的概率分布进行
        min_p=0.01,     # 最小概率阈值。 用于筛选出概率总和至少为min_p的词的集合，然后只从这个集合中进行采样
                        # 这种方法可以避免选择概率极低的词，同时又能保持一定的随机性，使得生成的文本更加合理和连贯
        skip_special_tokens=True,  # Whether to skip special tokens in the output跳过输出中的特殊token
        max_tokens=max_tokens,  # 生成文本的最大token数
        logit_bias={144540: -100, 21103: -100},  # logit_bias 是一个字典，用于对特定词的得分进行调整。字典的键是词的 ID，
                                                 # 值是要添加到该词的 logit（即模型输出的未经过 softmax 函数处理的得分）上的偏差值。
        # 通过调整 logit 值，可以影响词被选中的概率。例如，将某个词的 logit 减去一个较大的值（如 -100），
        # 会显著降低该词被选中的概率，甚至几乎不会被选中；而加上一个正值则会增加该词被选中的概率。
        stop=["</think>"],  # stop 是一个列表，其中包含的元素是停止生成的标记。当模型生成的文本中出现这些标记时，生成过程将立即停止
    )

    list_of_texts = [  # 将消息列表转换为文本列表
        tokenizer.apply_chat_template(  # 将消息列表转换为适合模型输入的文本格式
            conversation=messages,
            tokenize=False,  # 不进行token化
            add_generation_prompt=True  # 添加生成提示
        )
        for messages in list_of_messages  # 使用列表推导式遍历list_of_messages中的每个消息列表messages
    ]

    request_output = llm.generate(  # 使用模型生成回复
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    # 打印生成回复的token数量
    print([len(single_request_output.outputs[0].token_ids) for single_request_output in request_output])

    sort_keys_and_list_of_messages = []  # 用于存储生成回复的 token 数量和对应的消息列表
    # 将生成的回复添加到原始消息列表中
    for messages, single_request_output in zip(list_of_messages, request_output):
        # print()
        # print(single_request_output.outputs[0].text)
        # print() 将生成的回复以字典的形式添加到原始消息列表messages中

        # messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        messages += single_request_output.outputs[0].text
        # 将生成回复的 token 数量和更新后的消息列表作为元组添加到 sort_keys_and_list_of_messages 中
        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])  # 打印排序前列表中元组的第一个元素，即生成token数量
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])  # 根据生成token数量排序
    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])  # 打印排序后列表中元组的第一个元素

    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]  # 提取排序后的消息列表并返回

    return list_of_messages


def batch_message_filter(list_of_messages) -> tuple[list[list[dict]], list[str]]:
    """
    这段代码的主要功能是对一批消息进行处理，尝试从每组消息的最后一个消息内容中提取特定格式的文本（答案）。
    将成功提取到答案的情况记录在 extracted_answers 列表中，将没有提取到答案的消息组保留在 list_of_messages_to_keep 列表中，
    最后返回这两个列表组成的元组
    """
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers


# 方法1：
def create_starter_messages(question, index):
    """
    目的是根据给定的数学问题 question 和索引 index 生成一组对话消息列表，并根据索引选择其中一组消息返回。
    这些消息模拟了与数学助手进行交互的起始对话，为数学助手提供不同的系统提示信息，引导其按照特定的规则和思维方式解决给定的数学问题。
    """
    options = []
    for _ in range(13):
        options.append(
            [
                {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with "
                                              "deep resoning. You are careful and always recheck your conduction. "
                                              "You will never give answer directly until you have enough confidence. "
                                              "You should think step-by-step. Return final answer within \\boxed{}, "
                                              "after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
        )
    for _ in range(2):
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless math assistant. You should think "
                                              "step-by-step and you are good at reverse thinking to recheck your answer"
                                              " and fix all possible mistakes. After you get your final answer, "
                                              "take modulo 1000, and return the final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ],
        )
    options.append(
        [
            {"role": "system", "content": "Please carefully read the problem statement first to ensure you fully "
                                          "understand its meaning and key points. Then, solve the problem correctly "
                                          "and completely through deep reasoning. Finally, return the result modulo "
                                          "1000 and enclose it in \\boxed{} "
                                          "like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}."},
            {"role": "user", "content": question},
        ],
    )
    # 使用 index % len(options) 计算索引，确保索引在 options 列表的有效范围内
    return options[index%len(options)]


# 方法2
thought_prefix_english = """<think>
Alright, we have a math problem.
Hmm, it seems that I was asked to solve like a human. What does that mean? I guess I have to think through the problem 
step by step, similar to how a person would approach it. Think deeper. Humans work with easier numbers. 
They not do insane arithmetic. It means that when I have insane calculations to do, I am likely on the wrong track.
What else? This also means I should not be working with decimal places. I should avoid decimals.
Also, I should not submit answers that I am not sure."""


def create_starter_text(question: str, index: int) -> str:
    options = []
    for _ in range(7):
        messages = [
            {"role": "system",
             "content": "Solve the math problem from the user. Only submit an answer if you are sure. "
                        "Return final answer within \\boxed{}, after taking modulo 1000."},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>"
        options.append(starter_text)
    for _ in range(8):
        messages = [
            {"role": "system",
             "content": "Solve the math problem from the user, similar to how a human would (first think how would you "
                        "solve like a human). Only submit an answer if you are sure. After you get your final answer, "
                        "take modulo 1000, and return the final answer within \\boxed{}."},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + thought_prefix_english
        options.append(starter_text)
    for _ in range(1):
        messages = [
            {"role": "system", "content": "请通过逐步推理来解答问题，并把最终答案对1000取余数，放置于\\boxed{}中。"},
            {"role": "user", "content": question},
        ]
        starter_text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>"
        options.append(starter_text)

    return options[index % len(options)]


def predict_for_question(question: str) -> int:
    """
    根据输入的问题字符串 question 进行预测，并返回一个整数值作为预测结果。
    函数在执行过程中会根据不同的条件进行筛选和处理，调用其他辅助函数生成消息、过滤消息并最终选择一个答案
    """
    import os

    selected_questions_only = True      # 如果 selected_questions_only 为 True 且该环境变量不存在
    if selected_questions_only and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        if "Triangle" not in question:  # 会进一步检查问题字符串中是否包含特定的关键词,（如 "Triangle"、"delightful"、"George"）
            return 210                  # 如果不包含，则直接返回 210
        if "Triangle" not in question and "delightful" not in question and "George" not in question:
            return 210

    if time.time() > cutoff_time:
        return 210  # time.time() 用于获取当前时间戳。如果当前时间超过了 cutoff_time，则直接返回 210

    print(question)

    num_seqs = MAX_NUM_SEQS
    if time.time() > cutoff_times[-1]:  # 如果当前时间超过了 cutoff_times 列表中的最后一个时间戳，则将序列数量调整为原来的 2/3
        num_seqs = 2 * MAX_NUM_SEQS // 3
    # 生成 num_seqs 个消息列表: 每个消息列表由 create_starter_messages 函数根据问题和索引生成
    list_of_messages = [create_starter_messages(question, index) for index in range(num_seqs)]

    all_extracted_answers = []  # 用于存储所有提取到的答案
    for _ in range(1):          # 循环执行一次
        list_of_messages = batch_message_generate(list_of_messages)

        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):    # 如果环境变量 KAGGLE_IS_COMPETITION_RERUN 不存在
            df = pd.DataFrame(  # 将问题和每个消息列表的最后一个消息内容存储到一个 pandas 的 DataFrame 中，并将其保存为 CSV 文件
                {
                    "question": [question] * len(list_of_messages),
                    "message": [messages[-1]["content"] for messages in list_of_messages],
                }
            )                   # ，文件名根据当前时间与 start_time 的差值生成
            df.to_csv(f"{str(int(time.time() - start_time)).zfill(5)}.csv", index=False)
        # 对消息列表进行过滤，提取答案，并将提取到的答案添加到 all_extracted_answers 列表中
        list_of_messages, extracted_answers = batch_message_filter(list_of_messages)
        all_extracted_answers.extend(extracted_answers)

    print(all_extracted_answers)                    # 打印所有提取到的答案
    answer = select_answer(all_extracted_answers)   # 从 all_extracted_answers 列表中选择一个答案
    print(answer)

    print("\n\n")
    cutoff_times.pop()
    return answer


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:  # pl: polars
    """
    从输入的 id_ 和 question 两个 polars 数据框中提取第一个元素，将提取到的问题信息传入 predict_for_question 函数进行预测，
    得到预测答案后，将标识信息和预测答案组合成一个新的 polars 数据框并返回。同时，在过程中会打印一些信息用于调试和日志记录
    """
    id_ = id_.item(0)   # item() 方法用于从 polars 数据框中提取单个元素。这里从 id_ 数据框中提取第一个元素，将其赋值给 id_ 变量
    print("------")
    print(id_)

    question = question.item(0)
    answer = predict_for_question(question)
    print(question)
    print("------\n\n\n")
    return pl.DataFrame({'id': id_, 'answer': answer})


# 读取一个 CSV 文件，删除其中的 answer 列，然后将处理后的数据保存为一个新的 CSV 文件
pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')\
    .drop('answer', axis=1).to_csv('reference.csv', index=False)
# 测试
if os.getenv('KAGGLE_KERNEL_RUN_TYPE') == "Interactive" and not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    predict_for_question("Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. "
                         "Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. "
                         "What is the greatest possible length of segment $CD$?")

# 根据环境变量 KAGGLE_IS_COMPETITION_RERUN 的值来决定以不同的模式运行一个推理服务器。
# 推理服务器使用自定义的预测函数 predict 来处理输入数据并生成预测结果
# kaggle_evaluation.aimo_2_inference_server 是一个自定义的模块，推测它是为 Kaggle 竞赛中特定的 AI 数学奥林匹克相关任务开发的
# AIMO2InferenceServer 是该模块中定义的一个类，用于创建推理服务器
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
# 通过将 predict 函数传递给 AIMO2InferenceServer 的构造函数，将预测逻辑与推理服务器关联起来
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()    # 当环境变量 KAGGLE_IS_COMPETITION_RERUN 存在时，调用推理服务器的 serve 方法。
    # 推测这个方法会启动一个正式的服务，可能用于竞赛的重新运行阶段，以接收和处理来自竞赛平台的请求
else:
    inference_server.run_local_gateway( # KAGGLE_IS_COMPETITION_RERUN 不存在时，调用推理服务器的 run_local_gateway 方法。
        (                               # 这个方法接受一个元组作为参数，元组中包含一个或多个 CSV 文件的路径
#           '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
            'reference.csv',            # 推测这个方法会在本地模式下启动一个网关服务，使用指定的 CSV 文件作为输入数据进行预测
        )
    )