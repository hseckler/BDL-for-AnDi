#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class LSTM_Classification(nn.Module):
    '''
    An LSTM Network for classification tasks:
    the network consists of:
    3 stacked LSTM layers with adjustable hidden sizes (LSTM_size)
    1 Fully Connected layer with ReLU activation, of size hidden_size (default=20)
    1 Fully Connected output layer with no activation, outpute size corresponds to number of classes
    '''
    def __init__(self, num_input_features, num_classes, LSTM_size=64, hidden_size = 20):
        super(LSTM_Classification, self).__init__()
        self.num_input_features = num_input_features
        if type(LSTM_size) == type(64):
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size, 3, batch_first = True)
            self.dens = nn.Linear(LSTM_size,hidden_size)
            self.differentLSTM = False
        else:
            if len(LSTM_size) != 3:
                raise("LSTM size must be either int or array of 3 int!")
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size[0], 1, batch_first = True)
            self.LSTM2 = nn.LSTM(LSTM_size[0], LSTM_size[1], 1, batch_first = True)
            self.LSTM3 = nn.LSTM(LSTM_size[1], LSTM_size[2], 1, batch_first = True)
            self.dens = nn.Linear(LSTM_size[2],hidden_size)
            self.differentLSTM = True
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size,num_classes)
        
    #forward pass through the network
    def forward(self, x):
        out, hiddens = self.LSTM1(x)
        if self.differentLSTM:
            out, hiddens = self.LSTM2(out)
            out, hiddens = self.LSTM3(out)
        out = self.dens(out[:,-1,:])
        out = self.relu(out)
        out = self.fc(out)
        
        return out
    
class LSTM_Regression(nn.Module):
    '''
    An LSTM Network for regression tasks:
    the network consists of:
    3 stacked LSTM layers with adjustable hidden sizes (LSTM_size)
    1 Fully Connected output layer with no activation of output dimension
    '''
    def __init__(self, num_input_features, output_dim, LSTM_size=64):
        super(LSTM_Regression, self).__init__()
        self.num_input_features = num_input_features
        
        if type(LSTM_size) == type(64):
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size, 3, batch_first = True)
            self.fc = nn.Linear(LSTM_size,output_dim)
            self.differentLSTM = False
        else:
            if len(LSTM_size) != 3:
                raise("LSTM size must be either int or array of 3 int!")
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size[0], 1, batch_first = True)
            self.LSTM2 = nn.LSTM(LSTM_size[0], LSTM_size[1], 1, batch_first = True)
            self.LSTM3 = nn.LSTM(LSTM_size[1], LSTM_size[2], 1, batch_first = True)
            self.fc = nn.Linear(LSTM_size[2],output_dim)
            self.differentLSTM = True
            
    
    #forward pass through the network
    def forward(self, x):
        out, hiddens = self.LSTM1(x)
        if self.differentLSTM:
            out, hiddens = self.LSTM2(out)
            out, hiddens = self.LSTM3(out)
        out = self.fc(out[:,-1,:])
        
        return out
    
class LSTM_Regression_aleatoric(nn.Module):
    '''
    An LSTM Network for regression tasks:
    the network consists of:
    3 stacked LSTM layers with adjustable hidden sizes (LSTM_size)
    1 Fully Connected output layer with no activation of output dimension
    '''
    def __init__(self, num_input_features, output_dim=2, LSTM_size=64):
        super(LSTM_Regression_aleatoric, self).__init__()
        self.num_input_features = num_input_features
        if type(LSTM_size) == type(64):
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size, 3, batch_first = True)
            self.fc = nn.Linear(LSTM_size,output_dim)
            self.differentLSTM = False
        else:
            if len(LSTM_size) != 3:
                raise("LSTM size must be either int or array of 3 int!")
            self.LSTM1 = nn.LSTM(num_input_features, LSTM_size[0], 1, batch_first = True)
            self.LSTM2 = nn.LSTM(LSTM_size[0], LSTM_size[1], 1, batch_first = True)
            self.LSTM3 = nn.LSTM(LSTM_size[1], LSTM_size[2], 1, batch_first = True)
            self.fc = nn.Linear(LSTM_size[2],output_dim)
            self.differentLSTM = True
        self.outact = nn.Tanh()
            
    
    #forward pass through the network
    def forward(self, x):
        out, hiddens = self.LSTM1(x)
        if self.differentLSTM:
            out, hiddens = self.LSTM2(out)
            out, hiddens = self.LSTM3(out)
        out = self.fc(out[:,-1,:])
        out = self.outact(out)+1 #shape output to be between 0,2
        
        return out
