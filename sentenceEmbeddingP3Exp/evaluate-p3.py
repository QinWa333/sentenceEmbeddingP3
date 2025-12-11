import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
from prettytable import PrettyTable
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from senllm import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
from colorama import Fore, Style
import textwrap
from scipy.stats import spearmanr
import numpy as np
import yaml
import warnings
warnings.filterwarnings("ignore")


if torch.cuda.is_available():
    print("We are using GPU!")
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

# 设置日志
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# 设置路径
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# 导入SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def lock_and_write_file(file_path, content):
    with open(file_path, 'a') as file:
        while True:
            try:
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                file.write(content + '\n')
                file.flush()
            except IOError as e:
                print("File is locked by another process. Can't write.")
                time.sleep(1)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
                break

def load_config_from_yaml(config_file="config.yaml", config_name=None):
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"warning: config file {config_file} not found, using command line parameters")
        return None
    except yaml.YAMLError as e:
        print(f"error: config file {config_file} format error: {e}")
        return None
    
    if config_name is None:
        config_name = yaml_config.get('default_config', 'llama-2-7b')
    
    if config_name not in yaml_config.get('models', {}):
        available_configs = list(yaml_config.get('models', {}).keys())
        print(f"error: config '{config_name}' not found")
        print(f"available configs: {available_configs}")
        return None
    
    config = yaml_config['models'][config_name].copy()
    
    if 'gpu_config' in yaml_config:
        config['gpu_config'] = yaml_config['gpu_config']
    
    print(f"✓ successfully loaded config: {config_name}")
    return config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="配置名称，从config.yaml读取参数")
    parser.add_argument("--config_file", type=str, default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--model_name_or_path", type=str,
                        help="模型名称或路径")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="评估模式")
    parser.add_argument("--task_set", type=str,
                        choices=['sts', 'transfer', 'full', 'na', 'stsb'],
                        default='sts',
                        help="评估任务集")
    parser.add_argument("--num_placeholders", type=int, 
                        default=5,
                        help="插入的占位符数量")
    parser.add_argument("--output_layer", type=int, 
                        default=-1,
                        help="提取hidden states的层索引")
    parser.add_argument("--batch_size", type=int, 
                        default=16)

    args = parser.parse_args()
    
    if args.config:
        config = load_config_from_yaml(args.config_file, args.config)
        if config is None:
            print("config loading failed, exit program")
            sys.exit(1)
        
        args.model_name_or_path = config.get('model_name_or_path', args.model_name_or_path)
        args.output_layer = config.get('output_layer', args.output_layer)
        args.batch_size = config.get('batch_size', args.batch_size)
        args.mode = config.get('mode', args.mode)
        args.task_set = config.get('task_set', args.task_set)
        args.num_placeholders = config.get('num_placeholders', args.num_placeholders)
        
        if 'gpu_config' in config and 'cuda_visible_devices' in config['gpu_config']:
            os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_config']['cuda_visible_devices']
            print(f"✓ set GPU devices: {config['gpu_config']['cuda_visible_devices']}")
    
    if not args.model_name_or_path:
        print("error: model path not specified")
        sys.exit(1)
    
    hyper_parameters = textwrap.dedent(f"""
        {Fore.CYAN}Configuration:{Style.RESET_ALL}
        {Fore.YELLOW}-------------{Style.RESET_ALL}
        {Fore.GREEN}Backbone                :{Style.RESET_ALL} {args.model_name_or_path.split('/')[-1]}
        {Fore.GREEN}Method                  :{Style.RESET_ALL} P3-STS (Vanilla Mode)
        {Fore.GREEN}TP Status               :{Style.RESET_ALL} Disabled
        {Fore.GREEN}Number of Placeholders  :{Style.RESET_ALL} {args.num_placeholders}
        {Fore.GREEN}Output Layer Index      :{Style.RESET_ALL} {args.output_layer}
        {Fore.GREEN}Batch Size              :{Style.RESET_ALL} {args.batch_size}
    """)

    print(hyper_parameters)

    # 加载模型
    if 'llama' in args.model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        # 显式关闭TP功能，使用vanilla模式
        model.model.plan = 'vanilla'
        model.model.tp_starting_index = 0
        model.model.tp_exiting_index = 0
        print(f"{Fore.GREEN}✓ TP已关闭，使用vanilla模式{Style.RESET_ALL}")
    elif 'qwen2' in args.model_name_or_path.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        # 显式关闭TP功能，使用vanilla模式
        model.model.plan = 'vanilla'
        model.model.tp_starting_index = 0
        model.model.tp_exiting_index = 0
        print(f"{Fore.GREEN}✓ TP已关闭，使用vanilla模式{Style.RESET_ALL}")
    elif 'gemma' in args.model_name_or_path.lower():
        model = Gemma2ForCausalLM.from_pretrained(args.model_name_or_path,
                                                    device_map='auto',
                                                    output_hidden_states=True,
                                                    trust_remote_code=True)
        # 显式关闭TP功能，使用vanilla模式
        model.model.plan = 'vanilla'
        model.model.tp_starting_index = 0
        model.model.tp_exiting_index = 0
        print(f"{Fore.GREEN}✓ TP已关闭，使用vanilla模式{Style.RESET_ALL}")
    else:
        raise ValueError(f"Cannot find such {args.model_name_or_path.lower()} model!")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置评估任务
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        if args.mode == 'dev':
            args.tasks = ['STSBenchmark-dev']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'stsb':
        args.tasks = ['STSBenchmark']
    
    # 设置SentEval参数
    if args.mode == 'dev' or args.mode == 'fasttest':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':args.batch_size}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval的prepare和batcher函数
    def prepare(params, samples):
        return

    # 根据模型确定占位符token
    if 'llama' in args.model_name_or_path.lower():
        placeholder_token = '<unk>'
    else:
        placeholder_token = '<unk>'
    
    # 创建占位符字符串
    placeholder_string = placeholder_token * args.num_placeholders

    def batcher(params, batch, max_length=None):
        # 验证TP已关闭
        assert model.model.plan == 'vanilla', f"错误：期望vanilla模式，但当前是 {model.model.plan}"
        
        # 处理数据集中罕见的token编码问题
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        # 构建带占位符的提示词
        new_sentences = []
        for s in sentences:
            if len(s) > 0 and s[-1] not in '.?"\'': 
                s += '.'
            s = s.replace('"', '\'')
            if len(s) > 0 and '?' == s[-1]: 
                s = s[:-1] + '.'
            
            # 使用P3-STS提示词模板
            prompt = f'This sentence : \"{s}\" means in one word:\"{placeholder_string}'
            new_sentences.append(prompt)
        
        sentences = new_sentences

        batch_encoded = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # 将数据移动到正确的设备
        for k in batch_encoded:
            batch_encoded[k] = batch_encoded[k].to(device) if batch_encoded[k] is not None else None
        
        # 获取每个占位符位置的embedding
        with torch.no_grad():
            raw_outputs = model(output_hidden_states=True, return_dict=True, **batch_encoded)
            hidden_states = raw_outputs.hidden_states
            
            # 从指定层提取hidden states
            layer_hidden_states = hidden_states[args.output_layer]  # [batch_size, seq_len, hidden_dim]
            
            # P3方法：取占位符之前的最后一个token + 所有占位符token的hidden states
            # 参考p3-inference.py: ["logits"][0][-num_place_holders-1:]
            # 即取最后 (num_placeholders + 1) 个位置
            # 例如num_placeholders=5时，取位置: -6, -5, -4, -3, -2, -1
            embeddings_list = []
            
            for i in range(args.num_placeholders + 1):
                # 从后往前取：-(num_placeholders+1), -num_placeholders, ..., -2, -1
                position = -(args.num_placeholders + 1 - i)
                embedding = layer_hidden_states[:, position, :]  # [batch_size, hidden_dim]
                embeddings_list.append(embedding)
            
            # 堆叠并平均所有embedding
            # 注意：这里是 num_placeholders + 1 个位置
            stacked_embeddings = torch.stack(embeddings_list, dim=1)  # [batch_size, num_placeholders+1, hidden_dim]
            outputs = stacked_embeddings.mean(dim=1)  # [batch_size, hidden_dim]

            if outputs.dtype == torch.bfloat16:
                outputs = outputs.float()

            return outputs.cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark-dev']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        
        # 将结果写入文件
        if args.task_set != 'transfer':
            with open('./sts-p3-results', 'a') as f:
                model_name = args.model_name_or_path.split('/')[-1]
                f.write(f"{model_name} P3-{args.num_placeholders} " + ' '.join([str(s) for s in scores]) + '\n')

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

if __name__ == "__main__":
    main()
