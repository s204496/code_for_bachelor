import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# The NN argitecture
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN,self).__init__()
        self.bb_pre = nn.BatchNorm1d(input_size)
        self.in_layer = nn.Linear(input_size, 128)
        self.bb_1 = nn.BatchNorm1d(128)
        self.hid1 = nn.Linear(128, 256)
        self.bb_2 = nn.BatchNorm1d(256)
        self.hid2 = nn.Linear(256, 1024)
        self.hid3 = nn.Linear(1024, 1024)
        self.bb_3 = nn.BatchNorm1d(1024)
        self.hid4 = nn.Linear(1024, 256)
        self.hid5 = nn.Linear(256, 64)
        self.output = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        x128 = F.relu(self.in_layer(self.bb_pre(x))) # 128
        self.dropout
        x256_0 = F.relu(self.hid1(self.bb_1(x128))) # 256
        self.dropout
        x1024_0 = F.relu(self.hid2(self.bb_2(x256_0))) # 1024
        self.dropout
        x1024_1 = F.relu(self.hid3(self.bb_3(x1024_0)))
        self.dropout
        x1024_2 = F.relu(self.hid3(self.bb_3(x1024_1)))
        self.dropout
        x256_1 = F.relu(self.hid4(self.bb_3(x1024_2)))
        self.dropout
        x64 = F.relu(self.hid5(x256_1))
        x_end = self.output(x64)
        return x_end

#read and clean data for NN
def get_and_clean_data(df):
    n,d = df.shape
    for col in ['name', 'slug']:
        df[col] = df[col].astype(str).apply(lambda x: len(x))
    df.insert(loc=(df.columns.get_loc('slug'))+1, column='diff', value= (df['name'] - df['slug']))
    df['category'] = df['category'].astype(str).apply(lambda category: 0 if category == 'jam' else 1)
    df.insert(loc=(df.columns.get_loc('description'))+1, column='des_num_words', value= df['description'].astype(str).apply(lambda string: len(string.split())))
    df.insert(loc=(df.columns.get_loc('des_num_words'))+1, column='des_avg_words', value= df['description'].astype(str).apply(lambda string: 0.0 if len(string.split()) == 0 else sum(map(len, string.split()))/float(len(string.split()))))
    df['description'] = df['description'].astype(str).apply(lambda string: len(string))
 # df['published'] = df['published'].astype(str) see if its helpfull to do some date inclusion
 # see if a drop of version is helpfull
    for link in ['links', 'link-tags']:
        df[link] = df[link].apply(lambda string: 0 if string==0 else 1)
    df.drop(labels=['path', 'published', 'modified'], axis=1, inplace=True)
    return df

