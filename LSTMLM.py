import csv
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 另一个项目在用cuda 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 定义记录log
LOG_PATH = "./log_path"
SUB_FOLDER_PATH = LOG_PATH + "/training_100_epoch_torch_LSTM"
TRAINING_LOG_PATH = SUB_FOLDER_PATH + "/training.csv"
TESTING_LOG_PATH = SUB_FOLDER_PATH + "/testing.csv"
VALIDATION_LOG_PATH = SUB_FOLDER_PATH + "/validation.csv"
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
if not os.path.exists(SUB_FOLDER_PATH):
    os.mkdir(SUB_FOLDER_PATH)
# 写日志的file writer
training_log_headers = [" epoch ", " batch " ," loss " , " ppl "]
testing_log_headers = [" select model path " ," loss " , " ppl "]
validation_log_headers = [" samples after epoch " ," loss " , " ppl "]
training_file = open(TRAINING_LOG_PATH, 'a', newline='')
testing_file = open(TESTING_LOG_PATH, 'a', newline='')
validation_file = open(VALIDATION_LOG_PATH, 'a', newline='')
training_writer = csv.writer(training_file)
testing_writer = csv.writer(testing_file)
validation_writer = csv.writer(validation_file)
training_writer.writerow(training_log_headers)
testing_writer.writerow(testing_log_headers)
validation_writer.writerow(validation_log_headers)
save_model_base_path = "./models/torch_LSTM/"
if not os.path.exists(save_model_base_path):
    os.mkdir(save_model_base_path)

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8') #open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        if len(word) <= n_step:   #pad the sentence
            word = ["<pad>"]*(n_step+1-len(word)) + word

        for word_index in range(len(word)-n_step): # 得到n_step长度的数字，前5个对应第六个为target， input list
            input = [word2number_dict[n] for n in word[word_index:word_index+n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index+n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch # (batch num, batch size, n_step) (batch num, batch size)

def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  #open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))   #set to list

    word2number_dict = {w: i+2 for i, w in enumerate(word_list)}
    number2word_dict = {i+2: w for i, w in enumerate(word_list)}

    #add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict

class MyLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):# 输入embedding层的大小，输出隐藏层
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input 输入门 it
        self.Wii = torch.nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.Whi = torch.nn.Parameter(torch.Tensor(hidden_size,hidden_size))
        self.bi = torch.nn.Parameter(torch.Tensor(hidden_size))
        # forget 遗忘门 ft
        self.Wif = torch.nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.Whf = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = torch.nn.Parameter(torch.Tensor(hidden_size))
        # gt
        self.Wig = torch.nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.Whg = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = torch.nn.Parameter(torch.Tensor(hidden_size))
        # output 输出门 ot
        self.Wio = torch.nn.Parameter(torch.Tensor(input_size,hidden_size))
        self.Who = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = torch.nn.Parameter(torch.Tensor(hidden_size))
        # 初始化
        self.init_weight()

    def forward(self, X , init_states=None):
        # init_states传入h_t , c_t
        # X 形状 [n_step, batch_size, embedding size /input_size ],需要对每个单词
        n_step, batch_size, embedding_size = X.size()
        # 初始化h_t , c_t
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(X.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(X.device)
        else:
            h_t,c_t = init_states
        # 针对n_step中从第一个开始往后提取特征,c_t, h_t保留前面的信息作用到后面一个step对应的word上
        hidden_sequence = []
        for step_num in range(n_step):
            X_t = X[step_num,:,:] # batch size , embedding size # 128,256
            # print(X_t.size())
            i_t = torch.sigmoid(X_t @ self.Wii  + self.Whi @ h_t + self.bi)
            f_t = torch.sigmoid(X_t @ self.Wif  + self.Whf @ h_t + self.bf)
            g_t = torch.tanh(X_t @ self.Wig  + self.Whg @ h_t + self.bg)
            o_t = torch.sigmoid(X_t @ self.Wio + self.Who @ h_t + self.bo)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_sequence.append(h_t) # n_step, batch_size,hidden size (5,128,128)
        # print(h_t.size())
        hidden_sequence = torch.stack(hidden_sequence,0)
        # hidden_sequence = torch.cat(hidden_sequence, dim=0)
        # hidden_sequence = hidden_sequence.transpose(0, 1).contiguous()
        return hidden_sequence , (h_t,c_t) # 返回最后一次的结果

    def init_weight(self):
        # LSTM中的weight bias初始化为0
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        # self.LSTM = MyLSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False) # n class = 7615 n_hidden = 128
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)

        # hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # x:[128,5,256] - > [5,128,256]
        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]

        # outputs, (_, _) = self.LSTM(X, (hidden_state, cell_state)) # output (5,128,128)
        outputs, (_, _) = self.LSTM(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model

def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    
    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch) # input_batch : [batch_size, n_step] [128,5]

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item()) # 返回以e为底的x的指数
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))
                training_data_row = [epoch + 1] + [count_batch + 1]+[loss.item()] + [ppl]
                training_writer.writerow(training_data_row)
                training_file.flush()
            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))
        training_data_row = [epoch + 1] + [count_batch + 1] + [loss.item()] + [ppl]
        training_writer.writerow(training_data_row)
        training_file.flush()

        # valid after training one epoch ## get validation set and validate accuracy
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target)*128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1
          
            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))
            validation_data_row = [epoch + 1]+ [total_loss / count_loss] + [math.exp(total_loss / count_loss)]
            validation_writer.writerow(validation_data_row)
            validation_file.flush()
        if (epoch+1) % save_checkpoint_epoch == 0:
            torch.save(model, save_model_base_path+'LSTMlm_new_model_epoch'+str(epoch+1)+'.ckpt')

def test_LSTMlm(select_model_path):
    # model = torch.load(select_model_path, map_location="cpu")  #load the selected model
    model = torch.load(select_model_path, map_location="cpu")  #load the selected model
    model.to(device)

    #load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target)*128  # valid and test batch size is 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =','{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))
    test_data_row = [select_model_path] + [total_loss / count_loss] + [math.exp(total_loss / count_loss)]
    testing_writer.writerow(test_data_row)
    testing_file.flush()

if __name__ == '__main__':
    n_step = 5 # number of cells(= number of Step)
    n_hidden = 128 # number of hidden units in one cell
    batch_size = 128 # batch size
    learn_rate = 0.0005
    all_epoch = 100 #the all epoch for training
    emb_size = 256 #embeding size
    save_checkpoint_epoch = 5 # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt') # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    #print(word2number_dict)
    # word2number_dict : "符号“:数字  ； number2word_dict : 数字：“符号”

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  #n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]
    
    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)   #list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    # print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    for i in range(all_epoch):
        select_model_path = save_model_base_path+'LSTMlm_new_model_epoch'+str(i+1)+'.ckpt'
        if (i + 1) % save_checkpoint_epoch == 0:
            test_LSTMlm(select_model_path)
