import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt

class _2FC(nn.Module):
    def __init__(self, input_size=25, hidden_size=512):
        super(_2FC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.relu(x)
        x, _ = self.lstm2(x)
        x = self.fc2(x)
        return x

class Tokenizer():
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def get_dataloader(self, X_train, X_test, y_train, y_test, use_extra_features=False):      
        X_train_tensor = torch.stack([torch.tensor(i).float() for i in X_train])
        y_train_tensor = torch.tensor(y_train)

        X_test_tensor = torch.stack([torch.tensor(i).float() for i in X_test])
        y_test_tensor = torch.tensor(y_test)

        train = TensorDataset(X_train_tensor, y_train_tensor)
        test = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_sampler = RandomSampler(train)
        test_sampler = SequentialSampler(test)
        
        trainloader = DataLoader(train, sampler=train_sampler, batch_size=self.batch_size)
        testloader = DataLoader(test, sampler=test_sampler, batch_size=self.batch_size)

        return trainloader, testloader

class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, file_name='saved_weights', device='cpu', epochs=100):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.file_name = file_name

    def _train(self):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        total_preds=[]

        for step, batch in enumerate(self.train_dataloader):     
            batch = [r.to(self.device) for r in batch]
            sent_id, labels = batch
            self.model.zero_grad()

            preds = self.model(sent_id).view(-1)

            loss = self.loss_fn(preds, labels)
            total_loss = total_loss + loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            preds=preds.detach().cpu().numpy()
            total_preds.append(preds)
        
        avg_loss = total_loss / len(self.train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        return avg_loss, total_preds

    def _evaluate(self):
        self.model.eval()
        total_loss, total_accuracy = 0, 0
        total_preds=[]

        for step, batch in enumerate(self.val_dataloader):
            batch = [r.to(self.device) for r in batch]
            sent_id, labels = batch
            
            with torch.no_grad():
                preds = self.model(sent_id)
                loss = self.loss_fn(preds, labels)
                total_loss = total_loss + loss.item()
                preds=preds.detach().cpu().numpy()
                total_preds.append(preds)
            
        avg_loss = total_loss / len(self.train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
        return avg_loss, total_preds

    def save_score(self, file_name, score):
        with open(f"./nn/{file_name}.txt", "a") as f:
            f.write(f"{score}\n")

    def clear_file(self, file_name):
        with open(f"./nn/{file_name}.txt", "w") as f:
            f.write("")

    def train(self):
        best_valid_loss = float('inf')
        self.clear_file(f'{self.file_name}_train')
        self.clear_file(f'{self.file_name}_valid')

        for epoch in range(self.epochs):
            print(f'\n Epoch {epoch+1} / {self.epochs}')
            train_loss, _ = self._train()
            valid_loss, _ = self._evaluate()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'./nn/{self.file_name}.pt')

            self.save_score(f'{self.file_name}_train', train_loss)
            self.save_score(f'{self.file_name}_valid', valid_loss)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'\nValidation Loss: {valid_loss:.3f}')

    def test(self, X, y):
        X, y = torch.stack([torch.tensor(i).float() for i in X]), torch.tensor(y)

        with torch.no_grad():
            preds = self.model(X.to(self.device))
            preds = preds.detach().cpu().numpy()

        preds = torch.stack([torch.tensor(i[0]).float() for i in preds])

        report = mean_squared_error(y, preds)
        print('MSE:', report)

def prepare_data(crypto):
    df = pd.read_csv(f'data/{crypto}.csv')
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

    try:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    except:
        pass
    df.drop('open time', axis=1, inplace=True)
    df.drop('close time', axis=1, inplace=True)
    df.drop('quote asset volume', axis=1, inplace=True)
    df.drop('number of trades', axis=1, inplace=True)
    df.drop('tb base volume', axis=1, inplace=True)
    df.drop('tb quote volume', axis=1, inplace=True)
    df.drop('ignore', axis=1, inplace=True)

    X = df.to_numpy()[:-1]
    y = df['Close'][1:]
    X2, y2, X_tmp = [], [], []

    for idx, (i, j) in enumerate(zip(X, y)):
        if idx % 5 == 0 and idx != 0:
            X2 += [X_tmp]
            y2 += [j]
            X_tmp = []
        for x in i:
            X_tmp += [x]
    
    return X2, y2

def draw_chart(name):
    train = np.loadtxt(f'./nn/{name}_train.txt')
    valid = np.loadtxt(f'./nn/{name}_valid.txt')
    plt.plot(train, label='train')
    plt.plot(valid, label='valid')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'./nn/{name}.png')
    plt.clf()

def train_val_test_split(X, y):
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    index_60 = indices[:int(0.6*n)]

    X_train = [X[i] for i in index_60]
    y_train = [y[i] for i in index_60]

    X_tmp = [X[i] for i in indices if i not in index_60]
    y_tmp = [y[i] for i in indices if i not in index_60]

    n = len(X_tmp)
    indices = list(range(n))
    random.shuffle(indices)
    index_50 = indices[:int(0.5*n)]

    X_val = [X_tmp[i] for i in index_50]
    y_val = [y_tmp[i] for i in index_50]

    X_test = [X_tmp[i] for i in indices if i not in index_50]
    y_test = [y_tmp[i] for i in indices if i not in index_50]

    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_lite_data(crypto):
    df = pd.read_csv(f'data/{crypto}.csv')

    X = df['close'].to_list()[:-1]
    y = df['close'][1:]
    X2, y2 = [], []

    for i in range(len(X)):
        if i+4 >= len(X):
            break
        X2 += [[float(X[i]), float(X[i+1]), float(X[i+2]), float(X[i+3]), float(X[i+4])]]
        y2 += [float(y[i+4])]
    
    return X2, y2

# X, y = [], []

# for i in ['BTC', 'BNB', 'DOGE', 'DOT', 'ETH', 'GALA', 'HFT', 'XRP']:
#     X_tmp, y_tmp = prepare_lite_data(i)
#     X += X_tmp
#     y += y_tmp

# X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
# t = Tokenizer()
# train_dataloader, val_dataloader = t.get_dataloader(X_train, X_val, y_train, y_val)
# model = _2FC(input_size=5)

# trainer = Trainer(model, train_dataloader, val_dataloader, epochs=2500, file_name='model7')
# trainer.train()
# trainer.test(X_test, y_test)

# draw_chart('model7')