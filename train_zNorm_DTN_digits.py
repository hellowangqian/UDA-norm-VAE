import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy,scipy.io
from sklearn.preprocessing import normalize
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from models_zNorm_cnn import VAE,Classifier,Classifier_DTN
from sklearn.neighbors import KNeighborsClassifier
import torchvision
import loss
import pdb

domainSet =['mnist','usps','svhn']
train_transform=transforms.Compose([transforms.Resize(28),
                                transforms.Lambda(lambda x: x.convert("RGB")),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform=transforms.Compose([transforms.Resize(28),
                                transforms.Lambda(lambda x: x.convert("RGB")),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class TwoModalDataset(Dataset):
    def __init__(self,phase='train',sourceDomainIndex=0, targetDomainIndex = 0,trialIndex=0):
        data_dir = '../data/Digits/'
        self.datasets = {} 
        self.datasets['mnist-train'] = torchvision.datasets.MNIST(data_dir+'mnist/', download=True, train=True, transform=train_transform)
        self.datasets['mnist-test'] = torchvision.datasets.MNIST(data_dir+'mnist/', download=True, train=False, transform=test_transform)
        self.datasets['usps-train'] = torchvision.datasets.USPS(data_dir+'usps/', download=True, train=True, transform=train_transform)
        self.datasets['usps-test'] = torchvision.datasets.USPS(data_dir+'usps/', download=True, train=False, transform=test_transform)
        self.datasets['svhn-train'] = torchvision.datasets.SVHN(data_dir+'svhn/', download=False, split='train', transform=train_transform)
        self.datasets['svhn-test'] = torchvision.datasets.SVHN(data_dir+'svhn/', download=False, split='test', transform=test_transform)
        self.phase = phase

        self.sourceTrainDataset = self.datasets[domainSet[sourceDomainIndex]+'-train']
        self.num_class = 10#len(np.unique(self.sourceTrainDataset.targets))
        self.targetTrainDataset = self.datasets[domainSet[targetDomainIndex]+'-train']
        #self.targetLabels = self.targetTrainDataset.targets
        self.pseudoTargetLabels = np.ones_like((len(self.targetTrainDataset),))*-1 
        self.sourceTestDataset = self.datasets[domainSet[sourceDomainIndex]+'-test']
        self.testDataset = self.datasets[domainSet[targetDomainIndex]+'-test']
        self.update_dataset(self.pseudoTargetLabels)

    def __len__(self):
        if self.phase == 'train': #or self.phase == 'val':
            return len(self.sourceTrainDataset)
        if self.phase == 'target_train':
            return len(self.targetTrainDataset)
        if self.phase == 'test':
            return len(self.testDataset)
            
    def update_dataset(self,pseudo_label):
        self.pseudoTargetLabels = pseudo_label
        self.pseudoMask = self.pseudoTargetLabels!=-1
        self.pseudoLabelIndices = np.where(self.pseudoMask)[0]

    def __getitem__(self,idx):
        # return a pair of source and target images, which are from the same class but not necessarily the same image
        if self.phase == 'test':
            img = self.testDataset[idx][0]
            if img.shape[0] == 1:
                img = img.repeat(3,1,1)
            return img, self.testDataset[idx][1]
        elif self.phase == 'train':
            labelA = self.sourceTrainDataset[idx][1]
            imgA = self.sourceTrainDataset[idx][0]

            indicesB_this_label = np.argwhere(self.pseudoTargetLabels==labelA)
            if len(indicesB_this_label) > 0:
                idx_B = np.random.choice(indicesB_this_label[:,0])
                imgB = self.targetTrainDataset[idx_B][0]
                labelB = self.pseudoTargetLabels[idx_B]
            else:
                idx_B = np.random.randint(len(self.targetTrainDataset))
                imgB = self.targetTrainDataset[idx_B][0]
                labelB = np.ones_like(labelA)*-1
            if imgA.shape[0] == 1:
                imgA = imgA.repeat(3,1,1)
            if imgB.shape[0] == 1:
                imgB = imgB.repeat(3,1,1)
            
            return imgA,imgB,labelA,labelB
        elif self.phase == 'target_train':
            img = self.targetTrainDataset[idx][0]
            if img.shape[0] == 1:
                img = img.repeat(3,1,1)
            label = self.targetTrainDataset[idx][1]
            return img,label


def test_model(model,dataset,dataloader,device,model_type='knn'):
    since = time.time()
    num_class = dataset.num_class
    running_corrects = np.zeros((num_class,))
    num_sample_per_class = np.zeros((num_class,))
    # Iterate over data.
    for index, (features,labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            if model_type=='knn':
                preds = model.predict(features)
            if model_type=='mlp':
                model.eval()
                preds = model(features)
                preds = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
            if index == 0:
                outputs_test = preds
                labels_test = labels
            else:
                outputs_test = np.concatenate((outputs_test, preds), 0)
                labels_test = np.concatenate((labels_test, labels), 0)
        if model_type=='mlp':
            preds = np.argmax(outputs_test,1)
            scores = np.max(outputs_test,1)
        if model_type=='knn':
            preds = outputs_test
   
    for i in range(len(labels_test)):
        num_sample_per_class[labels_test[i]] += 1
        if preds[i]==labels_test[i]:
            running_corrects[labels_test[i]] += 1

    acc_per_class = running_corrects / num_sample_per_class
    print(f'per-image acc: {np.sum(running_corrects)/np.sum(num_sample_per_class) :2.4f}; ',end='')
    acc = np.mean(acc_per_class)
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('per-class acc: {:2.4f}'.format(acc))
    return preds, scores, acc_per_class,acc
        
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def loss_fn(recon_xS,recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch):
    criterion = torch.nn.MSELoss(size_average=False)
    mask = yT!=-1
    reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT[mask,:], xT[mask,:])
    cross_reconstruction_loss = criterion(recon_xS2[mask,:], xT[mask,:]) + criterion(recon_xT2[mask,:], xS[mask,:])
    #KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT[mask,:] - meanT[mask,:].pow(2) - log_varT[mask,:].exp())
    #distance = torch.sqrt(torch.sum((meanS[mask,:] - meanT[mask,:]) ** 2, dim=1) + torch.sum((torch.sqrt(log_varS[mask,:].exp()) - torch.sqrt(log_varT[mask,:].exp())) ** 2, dim=1))
    #distance = distance.sum()
    #weight = epoch*5e-4
    #print(f'{reconstruction_loss:1.4f}, {cross_reconstruction_loss:1.4f}')
    return (reconstruction_loss + cross_reconstruction_loss) / xS.size(0)

def train_classifier(classifier, vae, datasets, dataloaders, args, optimizer_cls, scheduler_cls):
    device = args.device
    vae.eval()
    acc_per_class = np.zeros((args.num_epochs_cls,datasets['train'].num_class))
    acc = np.zeros((args.num_epochs_cls,))
    max_iter = args.num_epochs_cls * len(datasets['train']) / args.batch_size
    iter_num = 0
    for epoch in range(args.num_epochs_cls):
        classifier.train()
        #print(f'Classifier training epoch {epoch:d}/{args.num_epochs_cls:d}')
        #print(optimizer_cls.param_groups[0]['lr'])
        for iteration, (xS,xT,yS,yT) in enumerate(dataloaders['train']):
            lr_scheduler(optimizer_cls, iter_num=iter_num, max_iter=max_iter)
            iter_num += 1
            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            recon_xS,recon_xT = generate_z(xS,xT,vae,device)
            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]          
            #pdb.set_trace()  
            recon_xT = recon_xT[mask,:]
            xtrain = torch.cat((xS,xT,recon_xS,recon_xT),dim=0)
            ytrain = torch.cat((yS,yT,yS,yT),dim=0)
            output = classifier(xtrain)
            #loss_cls = classifier.lossfunction(output, y)
            loss_cls = loss.CrossEntropyLabelSmooth(num_classes=10, epsilon=args.smooth)(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
        #scheduler_cls.step()
        #test_model(classifier,datasets['test'],dataloaders['test'], device,model_type='mlp')
    return classifier
    
def train_vae(vae, dataloader,args, optimizer, scheduler):
    ############################################################
    # train CVAE
    ############################################################
    device = args.device
    vae.train()
    for epoch in range(args.num_epochs_vae):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (xS,xT,yS,yT) in enumerate(dataloader):

            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, d=torch.zeros((xS.shape[0],1)).long().to(device))
            recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, d=torch.ones((xT.shape[0],1)).long().to(device))
            loss = loss_fn(recon_xS, recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return vae

############################################################
#Generating pseudo training samples and train/test a classifier
############################################################
def generate_z(xS,xT,vae,device):
    vae.eval()
    recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, d=torch.zeros((xS.shape[0],1)).long().to(device))
    recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, d=torch.ones((xT.shape[0],1)).long().to(device))
    return recon_xS2, recon_xT2
def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    ts = time.time()
    datasets = {x: TwoModalDataset(phase=x,sourceDomainIndex=args.sourceDomainIndex, targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex) for x in ['train', 'target_train', 'test']}
    dataloaders={}
    num_workers = 8
    # labeled source samples and pseudo labeled target samples
    dataloaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers = num_workers)
    # target training samples
    dataloaders['target_train'] = DataLoader(dataset=datasets['target_train'], batch_size=args.batch_size, shuffle=False, num_workers = num_workers)
    # target test samples
    dataloaders['test'] = DataLoader(dataset=datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers = num_workers)

    # define a classifier
    #classifier = Classifier_DTN(num_channels=3,num_classes=10).to(device) # train and test a classifier
    #optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=1e-2)
    #optimizer_cls = op_copy(optimizer_cls)
    #scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=20, gamma=0.5)
    acc_per_class = np.zeros((args.num_iter,10))

    # define the VAE
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        num_domains = 2).to(device)    
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=50, gamma=0.1)

    for iter in range(args.num_iter+5):
        if iter>0:
            # define VAE
            args.encoder_layer_sizes[0] = 4096
            vae = VAE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                num_domains = 2).to(device)    
            optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
            scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=50, gamma=0.1)
            
            # train VAE
            vae = train_vae(vae, dataloaders['train'], args, optimizer_vae, scheduler_vae)
        # train a classifier
        classifier = Classifier_DTN(num_channels=3,num_classes=10).to(device) # train and test a classifier
        #optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=1e-5)
        optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=1e-2)
        optimizer_cls = op_copy(optimizer_cls)
        scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=20, gamma=0.5)
        classifier = train_classifier(classifier, vae, datasets, dataloaders, args, optimizer_cls, scheduler_cls)    
        # classify target samples
        print(f'Iter {iter:02d}: ',end='')
        pseudo_labels, scores, acc_per_class, acc_per_image = test_model(classifier,datasets['target_train'],dataloaders['target_train'], device,model_type='mlp')
        test_model(classifier,datasets['test'],dataloaders['test'], device,model_type='mlp')
        # update pseudo_label_B,
        pseudo_label_B = -1*np.ones_like(pseudo_labels)
        trustable = np.zeros((len(pseudo_labels),),dtype=np.int32)
        numSelected = np.int32((iter+1)/args.num_iter*len(pseudo_labels)/10)
        for iCls in range(10):
            thisClassFlag = pseudo_labels==iCls
            numThisClass = thisClassFlag.sum()
            if numThisClass > 0:
                threshold = sorted(scores[thisClassFlag],reverse=True)[min(numThisClass-1,numSelected)]
                trustable = trustable + np.int32((scores>=threshold) & thisClassFlag)
        pseudo_label_B[trustable==1] = pseudo_labels[trustable==1]
        datasets['train'].update_dataset(pseudo_label_B)
        print((pseudo_label_B>-1).sum())
        #datasets['train'].pseudo_label_B[scores>0.9-iter*0.1] = pseudo_labels[scores>0.9-iter*0.1]        
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs_vae", type=int, default=20)
    parser.add_argument("--num_epochs_cls", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[800, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 800])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sourceDomainIndex", type=int, default=0)
    parser.add_argument("--targetDomainIndex", type=int, default=1)
    parser.add_argument("--trialIndex", type=int, default=0)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--num_iter", type=int, default=15)
    parser.add_argument('--smooth', type=float, default=0.1)  

    args = parser.parse_args()
    
    source = domainSet[args.sourceDomainIndex]
    target = domainSet[args.targetDomainIndex]
    args.filename = 'digits-'+source+'-'+target+'-trial'+str(args.trialIndex)+'-numIter-'+str(args.num_iter)+'-vaeEpochs-'+str(args.num_epochs_vae)+'-encoder_layer_sizes'+str(args.encoder_layer_sizes)+'-latSize-'+str(args.latent_size)+'-bs-'+str(args.batch_size)+'lr'+str(args.learning_rate)
    print(args.filename)
    main(args)
