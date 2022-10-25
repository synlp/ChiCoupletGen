import math
import os
from typing import List
from utils.format_metrics import compute_format_metrics
from utils.interactive_helper import parse_single
from utils.preprocess import PAD_token, UNK_token, EOS_token, SOS_token
from models.transformer import (
    make_decoder_dep_bertmodel,
    masked_cross_entropy,
    perplexity, 
    subsequent_mask)
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from measure.bleu import corpus_bleu

class DecoderDepBertTransformer(nn.Module):

    def __init__(self, path, tokenizer, bert=None, n_layers=3, 
                 head=4, 
                 d_ff=512, dropout=0.2, lr=0.001,
                 max_len=30):
        super(DecoderDepBertTransformer, self).__init__()
        assert path != '' or bert != None, "directory and bert cannot be empty at the same time."
        if path:
            self.model = torch.load(path)
        else:
            self.model = make_decoder_dep_bertmodel(bert=bert,
                                        N=n_layers,
                                        d_ff=d_ff,
                                        h=head,
                                        dropout=dropout)
        self.tokenizer = tokenizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.max_len = min(30, max_len)
        self.model.cuda()
        self.name = 'DecoderDepBertTransformer'
        self.freeze_encoder()
        # import pdb; pdb.set_trace()

    def save(self, path: str):
        torch.save(self.model, path)

    def freeze_encoder(self):
        """Freeze parameters of encoder and the shared embeddings"""
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)

    def release_encoder(self):
        """Release parameters of encoder and the shared embeddings"""
        for p in self.model.encoder.parameters():
            p.requires_grad_(True)

    def freeze_embeddings(self):
        for p in self.model.encoder.embeddings.parameters():
            p.requires_grad_(False)

    def release_embeddings(self):
        for p in self.model.encoder.embeddings.parameters():
            p.requires_grad_(True)

    def fit_batch(self, batch) -> float:
        # import pdb; pdb.set_trace()
        src = batch['src']
        tar = batch['tar']
        decoder_input = tar[:, :-1]
        tar = tar[:, 1:]
        src_mask = batch['src_mask']
        tar_mask = batch['tar_mask'][:, :-1].unsqueeze(-2)
        tar_mask = tar_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
        dep_text = batch['dep_text']
        dep_type = batch['dep_type']
        output_mask = (dep_text[:, :, 0] != self.tokenizer.pad_token_id)
        memory_mask = (dep_text != self.tokenizer.pad_token_id)
        # import pdb; pdb.set_trace()
        decoder_output, memory = self.model(src, decoder_input,
                                    src_mask, tar_mask, 
                                    dep_text, dep_type,
                                    memory_mask, output_mask)
        generator_input = torch.cat([decoder_output, memory[:, 1:, :]], dim=-1)
        log_probs = self.model.generator(generator_input)
        length = batch['tar_length']
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def fit(self, dataset):
        pbar = tqdm(dataset, total=len(dataset))
        tot_loss = 0
        num = 0
        jump_cnt = 0
        for batch in pbar:
            # loss = self.fit_batch(batch)
            # tot_loss += loss.item()
            # num += 1
            try:
                loss = self.fit_batch(batch)
                tot_loss += loss.item()
                num += 1
            except RuntimeError:
                jump_cnt += 1
            pbar.set_description(f'L: {tot_loss/num:<.2f}, skip {jump_cnt:2d} batch')

    def decode_batch(self, src: torch.Tensor, src_mask: torch.Tensor, dep_text, dep_type) -> List[str]:
        batch_size = src.size(0)
        decoder_input = torch.LongTensor([self.tokenizer.cls_token_id]*batch_size).cuda()
        decoder_input = decoder_input.unsqueeze(1)
        # import pdb; pdb.set_trace()
        for i in range(src.size(1)-1):
            tar_mask = (decoder_input != PAD_token)
            tar_mask = tar_mask.unsqueeze(-2)
            tar_mask = tar_mask & \
                subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
            output_mask = (dep_text[:, :, 0] != self.tokenizer.pad_token_id)
            memory_mask = (dep_text != self.tokenizer.pad_token_id)
            decoder_output, memory = self.model(src, decoder_input,
                                    src_mask, tar_mask, 
                                    dep_text, dep_type,
                                    memory_mask, output_mask)
            generator_input = torch.cat([decoder_output, memory[:, 1:i+2, :]], dim=-1)
            log_probs = self.model.generator(generator_input[:, -1, :])
            topv, topi = torch.topk(log_probs, 1)
            # topi = topi.squeeze(1)
            decoder_input = torch.cat([decoder_input, topi], dim=1)
        # decode to string
        decoded_str = []
        for i in range(batch_size):
            code = decoder_input[i, :].cpu()
            for j in range(1, code.size(-1)):
                if code[j] == self.tokenizer.sep_token_id:
                    decoded_str.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(
                        code[1:j].numpy().tolist()
                        )))
                    break
            else:
                decoded_str.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(
                    code[1:].numpy().tolist()
                    )))
        return decoded_str

    def batch_perplexity(self, batch) -> float:
        src = batch['src']
        tar = batch['tar']
        decoder_input = tar[:, :-1]
        tar = tar[:, 1:]
        src_mask = batch['src_mask']
        tar_mask = batch['tar_mask'][:, :-1].unsqueeze(-2)
        # import pdb; pdb.set_trace()
        tar_mask = tar_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
        dep_text = batch['dep_text']
        dep_type = batch['dep_type']
        output_mask = (dep_text[:, :, 0] != self.tokenizer.pad_token_id)
        memory_mask = (dep_text != self.tokenizer.pad_token_id)
        decoder_output, memory = self.model(src, decoder_input,
                                    src_mask, tar_mask, 
                                    dep_text, dep_type,
                                    memory_mask, output_mask)
        generator_input = torch.cat([decoder_output, memory[:, 1:, :]], dim=-1)
        log_probs = self.model.generator(generator_input)
        length = batch['tar_length']
        perplex = perplexity(log_probs.contiguous(), tar, length)
        return perplex

    def batch_loss(self, batch) -> float:
        src = batch['src']
        tar = batch['tar']
        decoder_input = tar[:, :-1]
        tar = tar[:, 1:]
        src_mask = batch['src_mask']
        tar_mask = batch['tar_mask'][:, :-1].unsqueeze(-2)
        # import pdb; pdb.set_trace()
        tar_mask = tar_mask & \
            subsequent_mask(decoder_input.size(-1)).type_as(tar_mask.data)
        dep_text = batch['dep_text']
        dep_type = batch['dep_type']
        output_mask = (dep_text[:, :, 0] != self.tokenizer.pad_token_id)
        memory_mask = (dep_text != self.tokenizer.pad_token_id)
        decoder_output, memory = self.model(src, decoder_input,
                                    src_mask, tar_mask, 
                                    dep_text, dep_type,
                                    memory_mask, output_mask)
        generator_input = torch.cat([decoder_output, memory[:, 1:, :]], dim=-1)
        log_probs = self.model.generator(generator_input)
        length = batch['tar_length']
        loss = masked_cross_entropy(log_probs.contiguous(), tar, length)
        return loss.item()

    def evaluate(self, dataset):
        decoded_output, tar_text, src_text = [], [], []
        tot_perplex, num = 0, 0
        # tot_loss = 0
        pbar = tqdm(dataset, total=len(dataset))
        # import pdb; pdb.set_trace()
        for batch in pbar:
            decoded_output += self.decode_batch(batch['src'], batch['src_mask'],
                                                batch['dep_text'], batch['dep_type'])
            text = batch['tar_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            tar_text += text
            text = batch['src_text']
            text = list(map(lambda s: ' '.join(list(s)), text))
            src_text += text
            tot_perplex += self.batch_perplexity(batch)
            # tot_loss += self.batch_loss(batch)
            num += 1
        perplexity = tot_perplex / num
        with open('hypothesis.txt', 'w', encoding='utf-8') as f:
            for out in decoded_output:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('reference.txt', 'w', encoding='utf-8') as f:
            for out in tar_text:
                f.write(''.join(out.split()))
                f.write('\n')
        with open('source.txt', 'w', encoding='utf-8') as f:
            for out in src_text:
                f.write(''.join(out.split()))
                f.write('\n')
        # import pdb; pdb.set_trace()
        total = 0
        format_correct = 0
        for out, tar in zip(decoded_output, tar_text):
            total += 1
            out_list, tar_list = out.split(), tar.split()
            if len(out_list) == len(tar_list):
                for out_ch, tar_ch in zip(out_list, tar_list):
                    if (tar_ch == '，' or out_ch == '，') and out_ch != tar_ch:
                        break
                else:
                    format_correct += 1
        format_acc = format_correct / total
        # import pdb; pdb.set_trace()
        tar_text = list(map(lambda s: [s], tar_text))
        bleu_score = corpus_bleu(decoded_output, tar_text)[0][0]*100
        print(f'BLEU-1: {corpus_bleu(decoded_output, tar_text, 1)}')
        print(f'BLEU-2: {corpus_bleu(decoded_output, tar_text, 2)}')
        print(f'BLEU-3: {corpus_bleu(decoded_output, tar_text, 3)}')
        print(f'BLEU score: {bleu_score:.4f}')
        print(f'Format acc: {format_acc:.4f}')
        print(f'Perplexity: {perplexity:.4f}')
        compute_format_metrics(src_text, decoded_output)
        # print(f'Loss      : {tot_loss/num:.2f}')
        return bleu_score, format_acc, perplexity

    def generate_response(self, src: str, wait=10) -> str:
        length = len(src)
        src_text = src
        src = self.tokenizer(src, return_tensors='pt')['input_ids']
        src_mask = self.tokenizer(src_text, return_tensors='pt')['attention_mask'].cuda()
        src = src.cuda()
        parse_result = parse_single(src_text, self.tokenizer)
        dep_text, dep_type, dep_memory_mask, dep_output_mask = parse_result['dep_text'], \
            parse_result['dep_type'], parse_result['dep_memory_mask'], parse_result['dep_output_mask']
        dep_text = dep_text.cuda()
        dep_type = dep_type.cuda()
        dep_memory_mask = dep_memory_mask.cuda()
        dep_output_mask = dep_output_mask.cuda()
        valid = False
        success = True
        decoded_str = []
        import time
        t0 = time.time()
        while not valid:
            if time.time() - t0 > wait:
                success = False
                break
            decoder_input = torch.tensor([[self.tokenizer.cls_token_id]]).cuda()
            for i in range(length+1):
                tar_mask = subsequent_mask(decoder_input.size(-1)).type(torch.bool).cuda()
                decoder_output, memory = self.model(src, decoder_input,
                                        src_mask, tar_mask, 
                                        dep_text, dep_type,
                                        dep_memory_mask, dep_output_mask)
                generator_input = torch.cat([decoder_output, memory[:, 1:i+2, :]], dim=-1)
                log_probs = self.model.generator(generator_input[:, -1, :])
                probs = torch.exp(log_probs)
                sampler = torch.distributions.categorical.Categorical(
                    probs=probs.squeeze()
                )
                index = sampler.sample().item()
                decoder_input = torch.cat([decoder_input, torch.tensor([[index]]).cuda()], dim=1)
            decoded_str = []
            # import pdb; pdb.set_trace()
            j = decoder_input.size(1) - 1
            if decoder_input[0, j] != self.tokenizer.sep_token_id:
                continue
            decoded_str = self.tokenizer.convert_ids_to_tokens(decoder_input[0, 1:j].cpu().numpy().tolist())
            for j in range(1, decoder_input.size(1)-1):
                if (src_text[j-1] == '，' or decoded_str[j-1] == '，') \
                        and src_text[j-1] != decoded_str[-1]:
                    break
                if decoder_input[0, j] == self.tokenizer.sep_token_id \
                        and j != length + 1:
                    break
            else:
                valid = True
        if not success: decoded_str = []
        return ''.join(decoded_str), success