from utils.preprocess import *
from models.DecoderSegPOSDepBertTransformer import DecoderSegPOSDepBertTransformer
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
dropout = 0.2
lr = 0.00001
max_epochs = 400
early_stopping = True
seg_epochs = 5
ratio = 0.1
window = 2

bert = BertModel.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
trn, dev, tst, human, max_len = preprocess_bert_seg_dep(tokenizer, batch_size, window, 'data')
human_seg = get_dataset_human_seg_dep('./data/segmentation', tokenizer, batch_size, window)

# with open('./save/lang.txt', 'wb') as f:
#     pickle.dump(lang, f)

mdl = DecoderSegPOSDepBertTransformer(path, tokenizer, bert, n_layers, 
                         head, d_ff, dropout, lr=lr,
                         max_len=max_len)
logging.info(f'Model type: {mdl.name}')
mdl.release_encoder()

for epoch in range(seg_epochs):
    logging.info(f'Learning for segmentation, epoch {epoch}')
    mdl.fit_seg(human_seg)

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
        mdl.save(f'./save/{mdl.name}-{bleu_score}-{format_acc}-{perplex}')
        print('model saved')
    if early_stopping:
        if bleu_score + format_acc > best:
            best = bleu_score + format_acc
            cnt = 0
            continue
        cnt += 1
        if cnt >= 5:
            print('Early Stopping...')
            break
