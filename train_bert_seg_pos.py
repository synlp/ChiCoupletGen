from utils.preprocess import *
from models.SegPOSBertTransformer import SegPOSBertTransformer
from transformers import BertTokenizer, BertModel
import logging
logging.basicConfig(level=logging.DEBUG)
# import pdb; pdb.set_trace()
batch_size = 32
path = ''
n_layers = 6
d_ff = 1024
head = 6
model_dim = 300
pos_dim = 32
dropout = 0.2
lr = 0.00001
max_epochs = 400
early_stopping = True
seg_epochs = 5
ratio = 0.1

bert = BertModel.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
# bert = BertModel.from_pretrained('./output', cache_dir='./output')
# tokenizer = BertTokenizer.from_pretrained('./output', cache_dir='./output')
trn, dev, tst, human, max_len = preprocess_bert_seg(tokenizer, batch_size, 'data')
human_seg = get_dataset_human_seg('./data/segmentation', tokenizer, batch_size)

# with open('./save/lang.txt', 'wb') as f:
#     pickle.dump(lang, f)

mdl = SegPOSBertTransformer(path, tokenizer, bert, n_layers, 
                         head, pos_dim, d_ff, dropout, lr=lr,
                         max_len=max_len, ratio=ratio)
logging.info(f'Model type: {mdl.name}')
mdl.release_encoder()

# for epoch in range(seg_epochs):
#     logging.info(f'Learning for segmentation, epoch {epoch}')
#     mdl.fit_seg(human_seg)

best = 0
cnt = 0
for epoch in range(max_epochs):
    print('epoch', epoch)
    mdl.train()
    mdl.fit(trn, human_seg)
    mdl.eval()
    with torch.no_grad():
        print("validating")
        bleu_score, format_acc, perplex = mdl.evaluate(dev)
        # mdl.save(f'./save/{mdl.name}(pretrain_again)-{bleu_score}-{format_acc}-{perplex}.pt')
        # logging.info(f'BLEU score: {bleu_score}')
        # logging.info(f'Format acc: {format_acc}')
        # logging.info(f'Perplexity: {perplex}')
        mdl.save(f'./save/{mdl.name}(no_pretrain_again)-{bleu_score}-{format_acc}-{perplex}')
        print('model saved')
    if early_stopping:
        if bleu_score + format_acc > best:
            best = bleu_score + format_acc
            cnt = 0
            continue
        cnt += 1
        if cnt >= 3:
            print('Early Stopping...')
            break
