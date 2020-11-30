import torch
import torch.nn as nn
import pdb
from utils import idx2onehot
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

class Classifier(nn.Module):
    def __init__(self,num_classes=10,num_channels=3):
        super(Classifier, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=100)
        self.drop = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=num_classes)
        
        self.lossfunction =  nn.CrossEntropyLoss()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def feature_extractor(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.drop(out)
        #out = self.fc2(out)    
        return out

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Classifier_DTN(nn.Module):
    def __init__(self,num_classes=10,num_channels=3):
        super(Classifier_DTN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(in_features=256*4*4, out_features=256)
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(256)
        self.fc2 = weightNorm(nn.Linear(in_features=256, out_features=num_classes))
        
        self.conv_layers.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        
        self.lossfunction =  nn.CrossEntropyLoss()
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

    def feature_extractor(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.drop(out)
        #out = self.fc2(out)    
        return out
        

        
class VAE(nn.Module):
    # One encoder one decoder, domain vector as input of decoder
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, device='cuda:0', num_labels=0,num_domains=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.device = device
        self.latent_size = latent_size
        self.encoder = Encoder2_1(
            encoder_layer_sizes, latent_size, conditional, num_labels, num_domains, device)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, num_domains, device)
        self.conditional = conditional
    def forward(self, x, c=None,d=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c, d)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        eps = eps.to(self.device)
        z = eps * std + means
        recon_x = self.decoder(z, c, d)
        recon_x2 = self.decoder(z, c, 1-d)
        
        if self.conditional:
            return recon_x,means,log_var,z
        return recon_x,recon_x2,means, log_var, z

    def inference(self, n=1, c=None, d=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c, d)

        return recon_x
        

class Encoder2_1(nn.Module):
    # For VAE2_1: domain information is not fed into the encoder
    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_domains,device):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.device = device
        ndf = 32
        nc = 3
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.fc1 = nn.Linear(in_features=1024, out_features=600)
        #self.drop = nn.Dropout2d(0.25)
        #self.fc2 = nn.Linear(in_features=600, out_features=120)
        #self.fc3 = nn.Linear(in_features=120, out_features=num_classes)

        self.linear_means = nn.Linear(600, latent_size)
        self.linear_log_var = nn.Linear(600, latent_size)

    def forward(self, x, c=None, d=None):

        x = self.encoder(x)
        x = self.fc1(x.view(-1, 1024))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        means = F.normalize(means, p=2, dim=1)
        log_vars = F.normalize(log_vars, p=2, dim=1)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_domains,device):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.device = device
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
        if self.num_domains > 0:
            input_size += num_domains
        ngf = 32
        nc = 3
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(   1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,      nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            #nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        if self.num_domains > 0:
            latent_size += num_domains
        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 1024)


    def forward(self, z, c=None, d=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)
        if self.num_domains > 0:
            d = idx2onehot(d.cpu(), n=self.num_domains).to(self.device)
            z = torch.cat((z,d), dim=-1)
        x = self.fc1(z)
        x = self.fc2(x)
        x = self.decoder(x.view(-1,1024,1,1))
        return x
