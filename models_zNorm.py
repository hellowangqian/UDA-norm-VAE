import torch
import torch.nn as nn
import pdb
from utils import idx2onehot
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self,input_dim, num_labels=65):
        super().__init__()
        self.fc = nn.Linear(input_dim,num_labels)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()
    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
        
class Classifier2(nn.Module):
    def __init__(self,input_dim, num_labels=65):
        super().__init__()
        self.fc = nn.Linear(input_dim,128)
        self.fc1 = nn.Linear(128,num_labels)
        self.logic = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.lossfunction =  nn.NLLLoss()
        #self.lossfunction =  nn.CrossEntropyLoss()
    def forward(self, x):
        o = self.logic(self.fc1(self.relu(self.fc(x))))
        #o = self.logic(self.fc(x))
        return o
        

class VAE(nn.Module):
    # One encoder one decoder, domain type as the last element of label indicator vector
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, num_domains=0, dropout=0):

        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, num_domains,dropout=dropout)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, num_domains,dropout=dropout)
  
    def forward(self, x, d=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, d)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to('cuda')
        z = eps * std + means
        recon_x = self.decoder(z, d)
        recon_x2 = self.decoder(z, 1-d)
        
        return recon_x,recon_x2,means, log_var, z

    def inference(self, n=1, d=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])
        recon_x = self.decoder(z, d)
        return recon_x
        

        
class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, num_domains,dropout=0):

        super().__init__()

        self.num_domains = num_domains
        if num_domains > 0:
            layer_sizes[0] = layer_sizes[0] + num_domains
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if dropout > 0:
                self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(dropout))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, d=None):

        if self.num_domains>0:
            d = idx2onehot(d.cpu(), n=self.num_domains).to('cuda')
            x = torch.cat((x, d), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        means = F.normalize(means, p=2, dim=1)
        log_vars = F.normalize(log_vars, p=2, dim=1)
        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, num_domains, device='cuda',dropout=0):

        super().__init__()

        self.MLP = nn.Sequential()

        self.num_domains = num_domains
        self.device = device
        input_size = latent_size
        if self.num_domains > 0:
            input_size = input_size + num_domains
            
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
                self.MLP.add_module(name="D{:d}".format(i), module=nn.Dropout(dropout))

            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z,d=None):

        if self.num_domains > 0:
            d = idx2onehot(d.cpu(), n=self.num_domains).to(self.device)
            z = torch.cat((z,d), dim=-1)
        x = self.MLP(z)
        x = F.normalize(x, p=2, dim=1)
        return x
