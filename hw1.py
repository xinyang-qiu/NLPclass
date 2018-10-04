
import glob
import numpy as np
import random
import spacy
import string
from tqdm import tqdm_notebook
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter



train_pos_txt = glob.glob("/Users/xinyangqiu/Downloads/aclImdb/train/pos/*.txt")
train_neg_txt = glob.glob("/Users/xinyangqiu/Downloads/aclImdb/train/neg/*.txt")
test_pos_txt = glob.glob("/Users/xinyangqiu/Downloads/aclImdb/test/pos/*.txt")
test_neg_txt = glob.glob("/Users/xinyangqiu/Downloads/aclImdb/test/neg/*.txt")


def read_txt(txt,ls):
    for fle in txt:
        with open(fle) as f:
            ls.append(f.read())
    return len(txt)
    
train_x = []
num_train_pos = read_txt(train_pos_txt,train_x)
num_train_neg = read_txt(train_neg_txt,train_x)
train_y = [1] * num_train_pos + [0] * num_train_neg

test_x = []
num_test_pos = read_txt(test_pos_txt,test_x)
num_test_neg = read_txt(test_neg_txt,test_x)
test_y = [1] * num_test_pos + [0] * num_test_neg


random.Random(4).shuffle(train_x)
random.Random(4).shuffle(train_y)

val_x = train_x[:5000]
val_y =train_y[:5000]
train_x = train_x[5000:]
train_y = train_y[5000:]

tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation
def tokenize(sent):
  tokens = tokenizer(sent)
  return [token.text.lower() for token in tokens if (token.text not in punctuations)]

def tokenizer_dt(dataset):
    token_dataset = []
    all_tokens = []

    for sample in tqdm_notebook(tokenizer.pipe(dataset, disable=['parser', 'tagger', 'ner'], batch_size=512, n_threads=1)):
        #tokens = lower_case_remove_punc(sample)
        tokens = [token.text.lower() for token in sample if (token.text not in punctuations)]
        token_dataset.append(tokens)
        all_tokens += tokens

    return token_dataset, all_tokens

token_train, all_train_tokens = tokenizer_dt(train_x)
token_test, all_test_tokens = tokenizer_dt(test_x)
token_val, all_val_tokens = tokenizer_dt(val_x)

max_vocab_size = 10000
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

token2id, id2token = build_vocab(all_train_tokens)


def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

train_data_indices = token2index_dataset(token_train)
test_data_indices = token2index_dataset(token_test)
val_data_indices = token2index_dataset(token_val)

MAX_SENTENCE_LENGTH = 200
class NewsGroupDataset(Dataset):
    def __init__(self, data_list, target_list):
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def newsgroup_collate_func(batch):
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]


BATCH_SIZE = 32
train_dataset = NewsGroupDataset(train_data_indices, train_y)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

val_dataset = NewsGroupDataset(val_data_indices, val_y)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=True)

test_dataset = NewsGroupDataset(test_data_indices, test_y)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=newsgroup_collate_func,
                                           shuffle=False)




class BagOfWords(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(BagOfWords, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)
    
    def forward(self, data, length):
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
        out = self.linear(out.float())
        return out



def test_model(loader, model):
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

emb_dim = 100
model = BagOfWords(len(id2token), emb_dim)

def output_model(emb_dim,learning_rate,num_epochs):
    model = BagOfWords(len(id2token), emb_dim)
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (data, lengths, labels) in enumerate(train_loader):
            model.train()
            data_batch, length_batch, label_batch = data, lengths, labels
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
        # validate every 100 iterations
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))
    return [emb_dim,learning_rate,num_epochs,test_model(val_loader, model),test_model(test_loader, model)]


learning_rate = [0.01,0.02]
emb_dim = [50,100,200,300,500]
num_epochs = [10,20,30,40,50]
result = []
for i in emb_dim:
    for j in learning_rate:
        for k in num_epochs:
            result.append(output_model(emb_dim = i,learning_rate = j,num_epochs = k))

result

