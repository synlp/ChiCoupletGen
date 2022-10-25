from typing import List, Tuple
import torch
import torch.utils.data as data
import logging

logging.basicConfig(level=logging.DEBUG)

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
SEP_token = 4

pos_list = ['SEP', 'NN', 'VV', 'AD', 'NR', 'PU', 'CD', 'M', 'JJ', # SEP is for EOS\SOS token
            'PN', 'VE', 'VA', 'BA', 'P', 'LC', 'VC', 'MSP',
            'NT', 'DT', 'CC', 'DEG', 'CS', 'DER', 'AS', 'OD',
            'SP', 'LB', 'SB', 'ETC', 'DEC', 'DEV', 'IJ', 'FW']
pos2id, id2pos = {}, {}
for i, tag in enumerate(pos_list):
    pos2id[tag] = i
    id2pos[i] = tag

class Lang:
    """As a dictionary"""

    def __init__(self):
        self.word2index = {'UNK': UNK_token,
                           'PAD': PAD_token,
                           'SOS': SOS_token,
                           'EOS': EOS_token,
                           '，':  SEP_token,
                           }
        self.index2word = {UNK_token: 'UNK',
                           PAD_token: 'PAD',
                           SOS_token: 'SOS',
                           EOS_token: 'EOS',
                           SEP_token: '，',
                           }
        self.vocab_size = 5
        print('Loading pretrained vectors')
        self.build_dict('sgns.sikuquanshu.word')
        print('Loaded')

    def build_dict(self, path):
        vectors = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            f.readline()
            for line in f:
                if ' ' not in line: continue
                word, s = line.split(maxsplit=1)
                if word in self.word2index:
                    continue
                self.word2index[word] = self.vocab_size
                self.index2word[self.vocab_size] = word
                self.vocab_size += 1
                vector = [eval(num) for num in s.strip('\n').split()]
                vectors.append(vector)
        vectors = torch.Tensor(vectors).cuda()
        self.vectors = vectors


    def add_words(self, words: str):
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = self.vocab_size
                self.index2word[self.vocab_size] = word
                self.vocab_size += 1

    def words2indices(self, words: str, tar=False):
        indices = [self.word2index.get(word, UNK_token) for word in words]
        if tar:
            return [SOS_token] + indices + [EOS_token]
        return indices


class Dataset:
    """For DataLoader"""

    def __init__(self, pairs, lang: Lang):
        self.pairs = pairs
        self.lang = lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        p['src'] = self.lang.words2indices(pair[0], False)
        p['tar'] = self.lang.words2indices(pair[1], True)
        p['tar_oneway'] = self.lang.words2indices(pair[1], False)
        p['tar_length'] = len(pair[1]) + 1   # EOS_token
        p['tar_text'] = pair[1]
        p['src_text'] = pair[0]
        return p

    def get_batch(self, pairs):

        def merge(seqs: List[List[int]]) -> torch.Tensor:
            max_len = max(list(map(len, seqs)))
            tensor = torch.ones((len(seqs), max_len)).long()
            for i in range(len(seqs)):
                tensor[i, :len(seqs[i])] = torch.tensor(seqs[i])
            return tensor

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = merge(ds['src'])
        tar = merge(ds['tar'])
        tar_oneway = merge(ds['tar_oneway'])
        src = src.cuda()
        tar = tar.cuda()
        tar_oneway = tar_oneway.cuda()
        batch = {'src': src, 'tar': tar,
                 'tar_oneway': tar_oneway,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch


class Dataset_bert:
    """For DataLoader"""

    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        p['tar_length'] = len(pair[1]) + 1   # EOS_token
        p['tar_text'] = pair[1]
        p['src_text'] = pair[0]
        return p

    def get_batch(self, pairs):
        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        src = src.cuda()
        tar = tar.cuda()
        src_mask = src_mask.cuda()
        tar_mask = tar_mask.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

class Dataset_bert_seg:
    """For DataLoader"""

    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def preprocess(self, raw: str):
        words = raw.split()
        text = ''
        pos = []
        for w in words:
            raw_text, pos_tag = w.split('_')
            text += raw_text
            pos += len(raw_text) * [pos2id[pos_tag]]
        return text, pos + [0]  # EOS(SEP)

    def __getitem__(self, index):
        pair = self.pairs[index]
        src_text, src_pos = self.preprocess(pair[0])
        tar_text, tar_pos = self.preprocess(pair[1])
        p = {}
        p['tar_length'] = len(tar_text) + 1   # EOS_token
        p['tar_text'] = tar_text
        p['src_text'] = src_text
        p['src_pos'] = src_pos
        p['tar_pos'] = tar_pos
        return p

    def get_batch(self, pairs):
        def merge(pos: List[List[int]]) -> torch.Tensor:
            max_len = max(map(len, pos))
            batch_size = len(pos)
            pos_tensor = torch.zeros((batch_size, max_len)).long()
            for i in range(batch_size):
                pos_tensor[i, :len(pos[i])] = torch.tensor(pos[i])
            return pos_tensor

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        src_pos = merge(ds['src_pos'])
        tar_pos = merge(ds['tar_pos'])
        src = src.cuda()
        tar = tar.cuda()
        src_mask = src_mask.cuda()
        tar_mask = tar_mask.cuda()
        src_pos = src_pos.cuda()
        tar_pos = tar_pos.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'src_pos': src_pos,
                 'tar_pos': tar_pos,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

class Dataset_human_seg:
    """For DataLoader"""

    def __init__(self, pairs_pos, pairs_seg, tokenizer):
        self.pairs_pos = pairs_pos
        self.pairs_seg = pairs_seg
        self.tokenizer = tokenizer
        self.seg_dict = {'B': 0, 'I': 1, 'E': 2, 'S': 3}

    def __len__(self):
        return len(self.pairs_pos)

    def preprocess(self, raw: str):
        words = raw.split()
        text = ''
        pos = []
        for w in words:
            raw_text, pos_tag = w.split('_')
            text += raw_text
            pos += len(raw_text) * [pos2id[pos_tag]]
        return text, pos + [0]  # EOS(SEP)

    def preprocess_seg(self, raw: str):
        words = raw.split()
        seg = []
        for word in words:
            if len(word) == 1:
                seg.append(self.seg_dict['S'])
                continue
            for i in range(len(word)):
                if i == 0:
                    seg.append(self.seg_dict['B'])
                elif i == len(word) - 1:
                    seg.append(self.seg_dict['E'])
                else: seg.append(self.seg_dict['I'])
        # CLS ... SEP
        return [self.seg_dict['S']] + seg + [self.seg_dict['S']]

    def __getitem__(self, index):
        pair_pos = self.pairs_pos[index]
        pair_seg = self.pairs_seg[index]
        src_text, src_pos = self.preprocess(pair_pos[0])
        tar_text, tar_pos = self.preprocess(pair_pos[1])
        src_seg = self.preprocess_seg(pair_seg[0])
        p = {}
        p['tar_length'] = len(tar_text) + 1   # EOS_token
        p['tar_text'] = tar_text
        p['src_text'] = src_text
        p['src_pos'] = src_pos
        p['tar_pos'] = tar_pos
        p['src_seg'] = src_seg
        return p

    def get_batch(self, pairs):
        def merge(pos: List[List[int]]) -> torch.Tensor:
            max_len = max(map(len, pos))
            batch_size = len(pos)
            pos_tensor = torch.zeros((batch_size, max_len)).long()
            for i in range(batch_size):
                pos_tensor[i, :len(pos[i])] = torch.tensor(pos[i])
            return pos_tensor

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        src_pos = merge(ds['src_pos'])
        tar_pos = merge(ds['tar_pos'])
        src_seg = merge(ds['src_seg'])
        src = src.cuda()
        tar = tar.cuda()
        src_mask = src_mask.cuda()
        tar_mask = tar_mask.cuda()
        src_pos = src_pos.cuda()
        tar_pos = tar_pos.cuda()
        src_seg = src_seg.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'src_pos': src_pos,
                 'tar_pos': tar_pos,
                 'src_seg': src_seg,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

def read_file(path: str) -> List[Tuple[str, str]]:
    max_len = 0
    with open(path + '/in.txt', 'r', encoding='utf-8') as f,\
            open(path + '/out.txt', 'r', encoding='utf-8') as g:
        cont = []
        for lin, lout in zip(f, g):
            cont.append((''.join(lin.strip('\n').split()), ''.join(lout.strip('\n').split())))
            max_len = max(max_len, len(lin.strip('\n')))
    return list(set(cont)), max_len

def read_file_seg(path: str) -> List[Tuple[str, str]]:
    max_len = 0
    with open(path + '/in_seg.txt', 'r', encoding='utf-8') as f,\
            open(path + '/out_seg.txt', 'r', encoding='utf-8') as g:
        cont = []
        for lin, lout in zip(f, g):
            cont.append((lin.strip('\n'), lout.strip('\n')))
            max_len = max(max_len, len(lin.strip('\n')))
    return cont, max_len

def read_file_human_seg(path: str) -> List[Tuple[str, str]]:
    max_len = 0
    with open(path + '/in.txt', 'r', encoding='utf-8') as f,\
            open(path + '/out.txt', 'r', encoding='utf-8') as g:
        cont = []
        for lin, lout in zip(f, g):
            cont.append((lin.strip('\n'), lout.strip('\n')))
            max_len = max(max_len, len(lin.strip('\n')))
    return cont, max_len

def get_dataset(path, lang: Lang, b_sz=32, shuffle=False):
    pairs, max_len = read_file(path)
    # if add_to_dict:
    #     for p in pairs:
    #         lang.add_words(p[0])
    #         lang.add_words(p[1])
    ds = Dataset(pairs, lang)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), \
        len(pairs), max_len

def get_bert_dataset(path, tokenizer, b_sz=32, shuffle=False):
    pairs, max_len = read_file(path)
    vocab  = tokenizer.get_vocab()
    # if add_to_dict:
    #     unk_list = []
    #     for p in pairs:
    #         for w in p[0]:
    #             if w not in vocab:
    #                 tokenizer.add_tokens(w)
    #                 vocab = tokenizer.get_vocab()
    #                 unk_list.append(w)
    #         for w in p[1]:
    #             if w not in vocab:
    #                 tokenizer.add_tokens(w)
    #                 vocab = tokenizer.get_vocab()
    #                 unk_list.append(w)
    #     print(unk_list)
    ds = Dataset_bert(pairs, tokenizer)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), len(pairs), max_len

def get_bert_dataset_seg(path, tokenizer, b_sz=32, shuffle=False):
    pairs, max_len = read_file_seg(path)
    ds = Dataset_bert_seg(pairs, tokenizer)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), len(pairs), max_len

def get_dataset_human_seg(directory, tokenizer, b_sz):
    pairs_pos, max_len = read_file_seg(directory)
    pairs_seg, max_len = read_file_human_seg(directory)
    logging.info(f'number of human segmentation samples {len(pairs_pos)}')
    logging.info(f'max length of human segmentation samples {max_len+1}')
    ds = Dataset_human_seg(pairs_pos, pairs_seg, tokenizer)
    return data.DataLoader(ds, b_sz, True, collate_fn=ds.get_batch)

def preprocess_bert(tokenizer, batch_size=16, data_dir='./data_set'):
    trn, trn_num, max_len_trn = get_bert_dataset(f'{data_dir}/trn', tokenizer, 
                                            b_sz=batch_size, shuffle=True)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_bert_dataset(f'{data_dir}/dev', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_bert_dataset(f'{data_dir}/tst', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {tst_num}')
    human, human_num, max_len_human = get_bert_dataset(f'{data_dir}/human', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {human_num}')
    max_len = max(max_len_trn, max_len_dev, max_len_tst, max_len_human) + 1 # EOS
    logging.info(f'number of words {len(tokenizer.get_vocab())}')
    logging.info(f'max length {max_len}')
    return trn, dev, tst, human, max_len

def preprocess_bert_seg(tokenizer, batch_size=16, data_dir='./data_set'):
    trn, trn_num, max_len_trn = get_bert_dataset_seg(f'{data_dir}/trn', tokenizer, 
                                            b_sz=batch_size, shuffle=True)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_bert_dataset_seg(f'{data_dir}/dev', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_bert_dataset_seg(f'{data_dir}/tst', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {tst_num}')
    human, human_num, max_len_human = get_bert_dataset_seg(f'{data_dir}/human', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {human_num}')
    max_len = max(max_len_trn, max_len_dev, max_len_tst, max_len_human) + 1 # EOS
    logging.info(f'number of words {len(tokenizer.get_vocab())}')
    logging.info(f'max length {max_len}')

    return trn, dev, tst, human, max_len


def preprocess(batch_size=16):
    lang = Lang()
    trn, trn_num, max_len_trn = get_dataset('./train', lang, 
                                            b_sz=batch_size, shuffle=True)
    logging.info(f'number of samples {trn_num}')
    # dev, dev_num, max_len_dev = get_dataset('./utils/dev.txt', lang, 
    #                                         b_sz=batch_size)
    dev, dev_num, max_len_dev = [], 0, 0
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_dataset('./test', lang, 
                                            b_sz=batch_size)
    max_len = max(max_len_trn, max_len_dev, max_len_tst) + 1 # EOS
    logging.info(f'number of samples {tst_num}')
    logging.info(f'number of words {lang.vocab_size}')
    logging.info(f'max length {max_len}')
    return trn, dev, tst, lang, max_len


############################################################################################
################## SECTION FOR DEPENDENCY PREPROCESS #######################################
############################################################################################

# null, mask are added for pointing to itself and special tokens (CLS, SEP)
dep_type_list = ['mask', 'null', 'name', 'nummod', 'mark:clf', 'nsubj', 'aux:modal', 
                 'dobj', 'compound:nn', 'case', 'advmod:loc', 'nmod:prep', 'dep', 
                 'nmod:topic', 'punct', 'conj', 'ccomp', 'amod', 'advcl:loc', 
                 'xcomp', 'advmod', 'nmod:assmod', 'cop', 'nmod:range', 'neg', 
                 'det', 'compound:vc', 'aux:ba', 'acl', 'nmod:tmod', 'appos', 
                 'advmod:rcomp', 'nmod', 'mark', 'amod:ordmod', 'cc', 'root', 
                 'aux:prtmod', 'aux:asp', 'parataxis:prnmod', 'nmod:poss', 
                 'advmod:dvp', 'etc', 'nsubjpass', 'auxpass', 'discourse', 'KILL']
dep2id, id2dep = {}, {}
for i, dep in enumerate(dep_type_list):
    dep2id[dep] = i
    id2dep[i] = dep

def parse_dep(lst: List[dict]):
    dep = {}
    id2words = {}
    for d in lst:
        governor_idx = d['governor']    # from index
        dependency = d['dep']           # dep type
        dependent_idx = d['dependent']  # to index
        governor = d['governorGloss']   # from word
        dependent = d['dependentGloss'] # to word
        if governor_idx not in dep:
            dep[governor_idx] = []
        dep[governor_idx].append((dependent_idx, dependency))
        id2words[governor_idx] = governor
        id2words[dependent_idx] = dependent
    # cut words into characters
    # import pdb; pdb.set_trace()
    id_word_pairs = sorted(list(id2words.items()))
    id2range = {}
    current_idx = 0
    text = ''
    for index, word in id_word_pairs[1:]:
        id2range[index] = (current_idx, current_idx+len(word))
        current_idx += len(word)
        text += word
    # import pdb; pdb.set_trace()
    char_dep = {}
    for gov_idx, dep_list in dep.items():
        if gov_idx == 0: continue
        for dep_idx, dep_type in dep_list:
            for i in range(*id2range[gov_idx]):
                for j in range(*id2range[dep_idx]):
                    if i not in char_dep: char_dep[i] = []
                    char_dep[i].append((text[j], dep_type))
    for i in range(len(text)):
        if i not in char_dep: char_dep[i] = []
        char_dep[i].append((text[i], 'null'))   # add itself
    return char_dep

def read_file_dep(directory):
    """return [(in, out, dependency)]"""
    max_len = 0
    with open(f'{directory}/in.txt', 'r', encoding='utf-8') as f, \
            open(f'{directory}/out.txt', 'r', encoding='utf-8') as g, \
            open(f'{directory}/in_dep.txt', 'r', encoding='utf-8') as h:
        cont = []
        for lin, lout, ldep in zip(f, g, h):
            cont.append((lin.strip(), lout.strip(), parse_dep(eval(eval(ldep.strip())))))
            max_len = max(max_len, len(lin.strip()))
    return cont, max_len

class Dataset_bert_dep:

    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        p['tar_length'] = len(pair[1]) + 1   # EOS_token
        p['tar_text'] = pair[1]
        p['src_text'] = pair[0]
        # import pdb; pdb.set_trace()
        p['dep_text'], p['dep_type'] = self.process_dep(pair[2], len(pair[0]))
        return p

    def process_dep(self, dep: dict, num: int):
        dep_text = [[] for _ in range(num)]
        dep_type_id = [[] for _ in range(num)]
        for i, lst in dep.items():
            for char, typ in lst:
                dep_text[i].append(char)
                dep_type_id[i].append(dep2id[typ])
        dep_text = dep_text
        dep_type_id = [[dep2id['mask']]] + dep_type_id + [[dep2id['mask']]]
        encoded = self.tokenizer(dep_text, is_split_into_words=True, add_special_tokens=False)
        dep_text_id = encoded['input_ids']
        dep_text_id_pad_special = [[self.tokenizer.pad_token_id]] + dep_text_id \
                                  + [[self.tokenizer.pad_token_id]]
        # assert len(dep_text_id_pad_special) == len(dep_type_id)
        # for text, type in zip(dep_text_id_pad_special, dep_type_id):
        #     assert len(text) == len(type), f'{text}, {type}'
        return dep_text_id_pad_special, dep_type_id
        

    def get_batch(self, pairs):

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

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        dep_text = merge(ds['dep_text'], pad=self.tokenizer.pad_token_id)
        dep_type = merge(ds['dep_type'], pad=dep2id['mask'])
        src = src.cuda()
        tar = tar.cuda()
        src_mask = src_mask.cuda()
        tar_mask = tar_mask.cuda()
        dep_text = dep_text.cuda()
        dep_type = dep_type.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'dep_text': dep_text,
                 'dep_type': dep_type,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

def get_bert_dep_dataset(directory, tokenizer, b_sz, shuffle=False):
    pairs, max_len = read_file_dep(directory)
    ds = Dataset_bert_dep(pairs, tokenizer)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), len(pairs), max_len

def preprocess_bert_dep(tokenizer, batch_size=16, data_dir='./data_set'):
    trn, trn_num, max_len_trn = get_bert_dep_dataset(f'{data_dir}/trn', tokenizer, 
                                            b_sz=batch_size, shuffle=True)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_bert_dep_dataset(f'{data_dir}/dev', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_bert_dep_dataset(f'{data_dir}/tst', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {tst_num}')
    human, human_num, max_len_human = get_bert_dep_dataset(f'{data_dir}/human', tokenizer, 
                                            b_sz=batch_size)
    logging.info(f'number of samples {human_num}')
    max_len = max(max_len_trn, max_len_dev, max_len_tst, max_len_human) + 1 # EOS
    logging.info(f'number of words {len(tokenizer.get_vocab())}')
    logging.info(f'max length {max_len}')
    return trn, dev, tst, human, max_len


############################################################################################
######################### SECTION FOR POS PREPROCESS #######################################
############################################################################################

class Dataset_bert_seg_memory:

    def __init__(self, pairs, tokenizer, window=2):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.window = window

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        src_text, pos_text, pos_type = self.parse_seg(pair[0])
        p['tar_text'] = ''.join([s.split('_')[0] for s in pair[1].split()])
        p['tar_length'] = len(p['tar_text']) + 1   # EOS_token
        p['src_text'] = src_text
        p['pos_text'] = pos_text
        p['pos_type'] = pos_type
        return p

    def parse_seg(self, seg: str):
        word_lst, pos_lst = [], []
        text = ''
        for s in seg.split():
            word, pos = s.split('_')
            word_lst.append(word)
            pos_lst.append(pos)
            text += word
        pos_text = [[] for _ in range(len(word_lst))]
        pos_type = [[] for _ in range(len(word_lst))]
        # import pdb; pdb.set_trace()
        for i in range(len(word_lst)):
            start = max(0, i-self.window)
            end = min(len(word_lst), i+self.window)
            pos_text[i] += word_lst[start:end+1]
            pos_type[i] += pos_lst[start:end+1]
        char_pos_text = []
        char_pos_type = []
        # import pdb; pdb.set_trace()
        for i in range(len(word_lst)):
            length = len(word_lst[i])
            char_text, char_type = [], []
            for word, pos in zip(pos_text[i], pos_type[i]):
                for char in word:
                    char_text.append(char)
                    char_type.append(pos)
            char_pos_text += [char_text]*length
            char_pos_type += [char_type]*length
        # convert to id
        # import pdb; pdb.set_trace()
        pos_type_id = []
        for i in range(len(char_pos_text)):
            char_pos_type_id = [pos2id[s] for s in char_pos_type[i]]
            pos_type_id.append(char_pos_type_id)
        # add CLS and SEP
        # import pdb; pdb.set_trace()
        pos_type_id = [[pos2id['SEP']]] + pos_type_id + [[pos2id['SEP']]]
        pos_text_id = self.tokenizer(char_pos_text, is_split_into_words=True,
                                     add_special_tokens=False)['input_ids']
        pos_text_id = [[self.tokenizer.pad_token_id]] + pos_text_id + \
            [[self.tokenizer.pad_token_id]]
        return text, pos_text_id, pos_type_id
        

    def get_batch(self, pairs):

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

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        pos_text = merge(ds['pos_text'], pad=self.tokenizer.pad_token_id)
        pos_type = merge(ds['pos_type'], pad=pos2id['SEP'])
        src = src.cuda()
        tar = tar.cuda()
        src_mask = src_mask.cuda()
        tar_mask = tar_mask.cuda()
        pos_text = pos_text.cuda()
        pos_type = pos_type.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'pos_text': pos_text,
                 'pos_type': pos_type,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

def get_bert_seg_memory_dataset(directory, tokenizer, b_sz, shuffle=False, window=2):
    pairs, max_len = read_file_seg(directory)
    ds = Dataset_bert_seg_memory(pairs, tokenizer, window=window)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), len(pairs), max_len

def preprocess_bert_seg_memory(tokenizer, batch_size=16, window=2, data_dir='./data_set'):
    trn, trn_num, max_len_trn = get_bert_seg_memory_dataset(f'{data_dir}/trn', tokenizer, 
                                            b_sz=batch_size, shuffle=True, window=window)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_bert_seg_memory_dataset(f'{data_dir}/dev', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_bert_seg_memory_dataset(f'{data_dir}/tst', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {tst_num}')
    human, human_num, max_len_human = get_bert_seg_memory_dataset(f'{data_dir}/human', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {human_num}')
    max_len = max(max_len_trn, max_len_dev, max_len_tst, max_len_human) + 1 # EOS
    logging.info(f'number of words {len(tokenizer.get_vocab())}')
    logging.info(f'max length {max_len}')
    return trn, dev, tst, human, max_len

############################################################################################
########################### SECTION FOR POS & DEPENDENCY PREPROCESS ########################
############################################################################################

class Dataset_bert_seg_dep:

    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        p = {}
        p['tar_text'] = pair[1]
        p['tar_length'] = len(p['tar_text']) + 1   # EOS_token
        p['src_text'] = pair[0]
        p['pos_text'] = self.process_pos_text(pair[2])
        p['pos_type'] = self.process_pos_type(pair[3])
        p['dep_text'], p['dep_type'] = self.process_dep(pair[4], len(pair[0]))
        return p

    def process_pos_text(self, pos_text: List[List[str]]):
        pos_text_id = self.tokenizer(pos_text, is_split_into_words=True, 
                                     add_special_tokens=False)['input_ids']
        pos_text_id = [[self.tokenizer.pad_token_id]] + pos_text_id + \
                        [[self.tokenizer.pad_token_id]]
        return pos_text_id

    def process_pos_type(self, pos_type: List[List[str]]):
        pos_type_id = []
        for i in range(len(pos_type)):
            pos_type_id.append([pos2id[s] for s in pos_type[i]])
        pos_type_id = [[pos2id['SEP']]] + pos_type_id + [[pos2id['SEP']]]
        return pos_type_id

    def process_dep(self, dep: dict, num: int):
        dep_text = [[] for _ in range(num)]
        dep_type_id = [[] for _ in range(num)]
        for i, lst in dep.items():
            for char, typ in lst:
                dep_text[i].append(char)
                dep_type_id[i].append(dep2id[typ])
        dep_text = dep_text
        dep_type_id = [[dep2id['mask']]] + dep_type_id + [[dep2id['mask']]]
        encoded = self.tokenizer(dep_text, is_split_into_words=True, add_special_tokens=False)
        dep_text_id = encoded['input_ids']
        dep_text_id_pad_special = [[self.tokenizer.pad_token_id]] + dep_text_id \
                                  + [[self.tokenizer.pad_token_id]]
        # assert len(dep_text_id_pad_special) == len(dep_type_id)
        # for text, type in zip(dep_text_id_pad_special, dep_type_id):
        #     assert len(text) == len(type), f'{text}, {type}'
        return dep_text_id_pad_special, dep_type_id

    def get_batch(self, pairs):

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

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        pos_text = merge(ds['pos_text'], pad=self.tokenizer.pad_token_id)
        pos_type = merge(ds['pos_type'], pad=pos2id['SEP'])
        dep_text = merge(ds['dep_text'], pad=self.tokenizer.pad_token_id)
        dep_type = merge(ds['dep_type'], pad=dep2id['mask'])
        if torch.cuda.is_available():
            src = src.cuda()
            tar = tar.cuda()
            src_mask = src_mask.cuda()
            tar_mask = tar_mask.cuda()
            pos_text = pos_text.cuda()
            pos_type = pos_type.cuda()
            dep_text = dep_text.cuda()
            dep_type = dep_type.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'pos_text': pos_text,
                 'pos_type': pos_type,
                 'dep_text': dep_text,
                 'dep_type': dep_type,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch

def parse_seg(seg: str, window: int):
    word_lst, pos_lst = [], []
    text = ''
    for s in seg.split():
        word, pos = s.split('_')
        word_lst.append(word)
        pos_lst.append(pos)
        text += word
    pos_text = [[] for _ in range(len(word_lst))]
    pos_type = [[] for _ in range(len(word_lst))]
    # import pdb; pdb.set_trace()
    for i in range(len(word_lst)):
        start = max(0, i-window)
        end = min(len(word_lst), i+window)
        pos_text[i] += word_lst[start:end+1]
        pos_type[i] += pos_lst[start:end+1]
    char_pos_text = []
    char_pos_type = []
    # import pdb; pdb.set_trace()
    for i in range(len(word_lst)):
        length = len(word_lst[i])
        char_text, char_type = [], []
        for word, pos in zip(pos_text[i], pos_type[i]):
            for char in word:
                char_text.append(char)
                char_type.append(pos)
        char_pos_text += [char_text]*length
        char_pos_type += [char_type]*length
    # add CLS and SEP
    return text, char_pos_text, char_pos_type

def read_file_seg_dep(directory, window=2):
    """return [(in, out, pos_text, pos_type, dependency)]"""
    max_len = 0
    with open(f'{directory}/in_seg.txt', 'r', encoding='utf-8') as f, \
            open(f'{directory}/out.txt', 'r', encoding='utf-8') as g, \
            open(f'{directory}/in_dep.txt', 'r', encoding='utf-8') as h:
        cont = []
        for lin, lout, ldep in zip(f, g, h):
            src_text, pos_text, pos_type = parse_seg(lin.strip(), window)
            cont.append((src_text, lout.strip(), pos_text, pos_type, parse_dep(eval(eval(ldep.strip())))))
            max_len = max(max_len, len(lin.strip()))
    return cont, max_len

def get_bert_seg_dep_dataset(directory, tokenizer, b_sz, shuffle=False, window=2):
    pairs, max_len = read_file_seg_dep(directory, window)
    ds = Dataset_bert_seg_dep(pairs, tokenizer)
    return data.DataLoader(ds, b_sz, shuffle, collate_fn=ds.get_batch), len(pairs), max_len

def preprocess_bert_seg_dep(tokenizer, batch_size=16, window=2, data_dir='./data_set'):
    trn, trn_num, max_len_trn = get_bert_seg_dep_dataset(f'{data_dir}/trn', tokenizer, 
                                            b_sz=batch_size, shuffle=True, window=window)
    logging.info(f'number of samples {trn_num}')
    dev, dev_num, max_len_dev = get_bert_seg_dep_dataset(f'{data_dir}/dev', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {dev_num}')
    tst, tst_num, max_len_tst = get_bert_seg_dep_dataset(f'{data_dir}/tst', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {tst_num}')
    human, human_num, max_len_human = get_bert_seg_dep_dataset(f'{data_dir}/human', tokenizer, 
                                            b_sz=batch_size, window=window)
    logging.info(f'number of samples {human_num}')
    max_len = max(max_len_trn, max_len_dev, max_len_tst, max_len_human) + 1 # EOS
    logging.info(f'number of words {len(tokenizer.get_vocab())}')
    logging.info(f'max length {max_len}')
    logging.info(f'window: {window}')
    return trn, dev, tst, human, max_len

###########################################################################################
####################### SECTION FOR SEGMENTATION & POS & DEPENDENCY #######################
###########################################################################################

class Dataset_human_seg_dep(Dataset_bert_seg_dep):
    """For DataLoader"""

    def __init__(self, pairs_pos_dep, pairs_seg, tokenizer):
        self.pairs_pos_dep = pairs_pos_dep
        self.pairs_seg = pairs_seg
        self.tokenizer = tokenizer
        self.seg_dict = {'B': 0, 'I': 1, 'E': 2, 'S': 3}

    def __len__(self):
        return len(self.pairs_pos_dep)

    def preprocess_seg(self, raw: str):
        words = raw.split()
        seg = []
        for word in words:
            if len(word) == 1:
                seg.append(self.seg_dict['S'])
                continue
            for i in range(len(word)):
                if i == 0:
                    seg.append(self.seg_dict['B'])
                elif i == len(word) - 1:
                    seg.append(self.seg_dict['E'])
                else: seg.append(self.seg_dict['I'])
        # CLS ... SEP
        return [self.seg_dict['S']] + seg + [self.seg_dict['S']]

    def __getitem__(self, index):
        p = {}
        pair_pos_dep = self.pairs_pos_dep[index]
        pair_seg = self.pairs_seg[index]
        p['tar_text'] = ''.join(pair_pos_dep[1].split())
        p['tar_length'] = len(p['tar_text']) + 1   # EOS_token
        p['src_text'] = pair_pos_dep[0]
        p['pos_text'] = self.process_pos_text(pair_pos_dep[2])
        p['pos_type'] = self.process_pos_type(pair_pos_dep[3])
        p['dep_text'], p['dep_type'] = self.process_dep(pair_pos_dep[4], len(pair_pos_dep[0]))
        src_seg = self.preprocess_seg(pair_seg[0])
        p['src_seg'] = src_seg
        return p

    def get_batch(self, pairs):

        def merge_(pos: List[List[int]]) -> torch.Tensor:
            max_len = max(map(len, pos))
            batch_size = len(pos)
            pos_tensor = torch.zeros((batch_size, max_len)).long()
            for i in range(batch_size):
                pos_tensor[i, :len(pos[i])] = torch.tensor(pos[i])
            return pos_tensor

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

        ds = {}
        for key in pairs[0].keys():
            ds[key] = []
        for p in pairs:
            for key in p.keys():
                ds[key].append(p[key])
        src = self.tokenizer(ds['src_text'], padding=True, return_tensors='pt')
        tar = self.tokenizer(ds['tar_text'], add_special_tokens=True, 
                             padding=True, return_tensors='pt')
        src_mask = src['attention_mask']
        src = src['input_ids']
        tar_mask = tar['attention_mask']
        tar = tar['input_ids']
        pos_text = merge(ds['pos_text'], pad=self.tokenizer.pad_token_id)
        pos_type = merge(ds['pos_type'], pad=pos2id['SEP'])
        dep_text = merge(ds['dep_text'], pad=self.tokenizer.pad_token_id)
        dep_type = merge(ds['dep_type'], pad=dep2id['mask'])
        src_seg = merge_(ds['src_seg'])

        if torch.cuda.is_available():
            src = src.cuda()
            tar = tar.cuda()
            src_mask = src_mask.cuda()
            tar_mask = tar_mask.cuda()
            pos_text = pos_text.cuda()
            pos_type = pos_type.cuda()
            dep_text = dep_text.cuda()
            dep_type = dep_type.cuda()
            src_seg = src_seg.cuda()
        batch = {'src': src, 'tar': tar,
                 'src_mask': src_mask,
                 'tar_mask': tar_mask,
                 'pos_text': pos_text,
                 'pos_type': pos_type,
                 'dep_text': dep_text,
                 'dep_type': dep_type,
                 'src_seg': src_seg,
                 'tar_length': ds['tar_length'],
                 'tar_text': ds['tar_text'],
                 'src_text': ds['src_text'],
                 }
        return batch


def get_dataset_human_seg_dep(directory, tokenizer, b_sz, window=2):
    pairs_pos_dep, max_len = read_file_seg_dep(directory, window)
    pairs_seg, max_len = read_file_human_seg(directory)
    logging.info(f'number of human segmentation samples {len(pairs_pos_dep)}')
    logging.info(f'max length of human segmentation samples {max_len+1}')
    logging.info(f'window: {window}')
    ds = Dataset_human_seg_dep(pairs_pos_dep, pairs_seg, tokenizer)
    return data.DataLoader(ds, b_sz, True, collate_fn=ds.get_batch)


if __name__ == '__main__':
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
    # tokenizer.add_special_tokens({'bos_token': '<BOS>', 'eos_token': '<EOS>'})
    seg = get_dataset_human_seg_dep('./data/segmentation', tokenizer, 4)
    # trn, dev, tst, human, max_len = preprocess_bert_seg_dep(tokenizer, 4, data_dir='./data')
    # human, _, _ = get_bert_seg_dep_dataset('./data/human', tokenizer, 4)
    for i, batch in enumerate(seg):
        assert batch['pos_text'].shape == batch['pos_type'].shape, f"{batch['src_text']}, {batch['tar_text']}, {batch['pos_text']}, {batch['pos_type']}"
        if i == 0:
            print(batch)
            import pdb; pdb.set_trace()
            # break
