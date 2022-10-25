from utils.preprocess import *
from models.BertTransformer import BertTransformer
from transformers import BertTokenizer, BertModel
import pickle
# import pdb; pdb.set_trace()
batch_size = 16
path = ''
n_layers = 6
d_ff = 1024
head = 6
model_dim = 300
dropout = 0.2
lr = 0.00001
max_epochs = 400
early_stopping = True

bert = BertModel.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-chinese', cache_dir='./pretrained')
trn, dev, tst, human, max_len = preprocess_bert(tokenizer, batch_size, 'data')

# with open('./save/lang.txt', 'wb') as f:
#     pickle.dump(lang, f)

mdl = BertTransformer(path, tokenizer, bert, n_layers, 
                         head, d_ff, dropout, lr=lr,
                         max_len=max_len)
mdl.release_encoder()

best = 0
cnt = 0
for epoch in range(max_epochs):
    print('epoch', epoch)
    mdl.train()
    mdl.fit(trn)
    mdl.eval()
    with torch.no_grad():
        print("validating")
        bleu_score, format_acc, perplex = mdl.evaluate(dev)
        mdl.save(f'./save/BertFineTune-{bleu_score}-{format_acc}-{perplex}.pt')
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
