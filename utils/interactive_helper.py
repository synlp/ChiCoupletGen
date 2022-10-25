import os
import torch
import tempfile
from typing import List
from utils.preprocess import parse_dep, parse_seg
from utils.preprocess import pos2id, dep2id

def process_pos_text(pos_text: List[List[str]], tokenizer):
    pos_text_id = tokenizer(pos_text, is_split_into_words=True, 
                                 add_special_tokens=False)['input_ids']
    pos_text_id = [[tokenizer.pad_token_id]] + pos_text_id + \
                    [[tokenizer.pad_token_id]]
    return pos_text_id

def process_pos_type(pos_type: List[List[str]]):
    pos_type_id = []
    for i in range(len(pos_type)):
        pos_type_id.append([pos2id[s] for s in pos_type[i]])
    pos_type_id = [[pos2id['SEP']]] + pos_type_id + [[pos2id['SEP']]]
    return pos_type_id

def process_dep(dep: dict, num: int, tokenizer):
    dep_text = [[] for _ in range(num)]
    dep_type_id = [[] for _ in range(num)]
    for i, lst in dep.items():
        for char, typ in lst:
            dep_text[i].append(char)
            dep_type_id[i].append(dep2id[typ])
    dep_text = dep_text
    dep_type_id = [[dep2id['mask']]] + dep_type_id + [[dep2id['mask']]]
    encoded = tokenizer(dep_text, is_split_into_words=True, add_special_tokens=False)
    dep_text_id = encoded['input_ids']
    dep_text_id_pad_special = [[tokenizer.pad_token_id]] + dep_text_id \
                              + [[tokenizer.pad_token_id]]
    return dep_text_id_pad_special, dep_type_id

def merge(batch: List[List[List[int]]], pad=0):
    batch_size = len(batch)
    max_sentence_len = max([len(lst) for lst in batch])
    max_memory_len = 0
    for sentence in batch:
        for mem in sentence:
            max_memory_len = max(max_memory_len, len(mem))
    ten = torch.zeros((batch_size, max_sentence_len, max_memory_len)).fill_(pad).long()
    for i, sentence in enumerate(batch):
        for j, memory in enumerate(sentence):
            ten[i, j, :len(memory)] = torch.tensor(memory)
    return ten

def parse_single(sentence: str, tokenizer, window=2):
    raw_pos, raw_dep = parse_raw(sentence)
    results = {}
    src_text, pos_text, pos_type = parse_seg(raw_pos, window)
    pos_text = process_pos_text(pos_text, tokenizer)
    pos_type = process_pos_type(pos_type)
    dep = parse_dep(raw_dep)
    dep_text, dep_type = process_dep(dep, len(sentence), tokenizer)
    pos_text = merge([pos_text])
    pos_type = merge([pos_type])
    dep_text = merge([dep_text])
    dep_type = merge([dep_type])
    dep_output_mask = (dep_text[:, :, 0] != tokenizer.pad_token_id)
    dep_memory_mask = (dep_text != tokenizer.pad_token_id)
    pos_output_mask = (pos_text[:, :, 0] != tokenizer.pad_token_id)
    pos_memory_mask = (pos_text != tokenizer.pad_token_id)
    results = {
        'pos_text': pos_text,
        'pos_type': pos_type,
        'pos_output_mask': pos_output_mask,
        'pos_memory_mask': pos_memory_mask,
        'dep_text': dep_text,
        'dep_type': dep_type,
        'dep_output_mask': dep_output_mask,
        'dep_memory_mask': dep_memory_mask,
    }
    return results

def parse_raw(sentence):
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f'{tmpdir}/in.txt', 'w') as f:
            f.write(sentence)
        cmd = f'bash run_sample.sh in.txt in_seg.txt {tmpdir}'
        os.system(cmd)
        with open(f'{tmpdir}/in_seg.txt', 'r') as f:
            raw_pos = f.read().strip()
        with open(f'{tmpdir}/in.stanford.json', 'r') as f:
            raw_dep = eval(f.read().strip())['basicDependencies']
    return raw_pos, raw_dep

if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
    # print(parse_raw('观音山上观山水'))
    print(parse_single('观音山上观山水', tokenizer, 2))