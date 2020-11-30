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
from models_zNorm import VAE,Classifier
from sklearn.neighbors import KNeighborsClassifier
import pdb

domainSet =['i','p','c']
class TwoModalDataset(Dataset):
    def __init__(self,phase='train',sourceDomainIndex=0, targetDomainIndex = 0,trialIndex=0):
        self.phase = phase
        self.load_mat(sourceDomainIndex,targetDomainIndex,trialIndex)
        self.pseudo_label_B = np.ones_like(self.label_B)*-1 # this will be dynamically updated during training
        self.pseudo_score_B = np.zeros_like(self.label_B)
    def load_mat(self,sourceDomainIndex=0, targetDomainIndex=0,trialIndex=0):
        # load features and labels
        data_dir = '../data/image-clef/'
        # data_dir = 'E:\DomainAdaptation\OfficeHomeDataset_10072016/'
        data_A = scipy.io.loadmat(data_dir+'imageclef-'+domainSet[sourceDomainIndex]+'-resnet50-noft.mat')
        feature_A = data_A['resnet50_features'][:,:,0,0]
        self.feature_A = normalize(feature_A,norm='l2')
        self.label_A = data_A['labels'][0,]
        self.num_class = len(np.unique(self.label_A))
        data_B = scipy.io.loadmat(data_dir+'imageclef-'+domainSet[targetDomainIndex]+'-resnet50-noft.mat')
        feature_B = data_B['resnet50_features'][:,:,0,0]
        self.feature_B = normalize(feature_B,norm='l2')
        self.label_B = data_B['labels'][0,]
        
    def __len__(self):
        if self.phase == 'train': #or self.phase == 'val':
            return self.feature_A.shape[0]
        if self.phase == 'test':
            return self.feature_B.shape[0]
    def __getitem__(self,idx):
        if self.phase == 'test':
            idx_B = idx
            return self.feature_B[idx_B,:],self.label_B[idx_B]
        # return a pair of regular and xray image features, which are paired randomly
        label = self.label_A[idx]
        #indicesB_this_label = np.argwhere((self.pseudo_label_B==label) & (self.pseudo_score_B > -1))
        indicesB_this_label = np.argwhere(self.pseudo_label_B==label)
        if len(indicesB_this_label) > 0:
            idx_B = np.random.choice(indicesB_this_label[:,0])
            return self.feature_A[idx,:], self.feature_B[idx_B,:],self.label_A[idx],self.pseudo_label_B[idx_B]
        else:
            idx_B = np.random.randint(len(self.label_B))
            return self.feature_A[idx,:], self.feature_B[idx_B,:], self.label_A[idx], np.ones_like(self.label_A[idx]) * -1


def test_model(model,dataset,dataloader,device,model_type='knn'):
    since = time.time()
    
    num_class = dataset.num_class
    running_corrects = np.zeros((num_class,))
    num_sample_per_class = np.zeros((num_class,))
    # Iterate over data.
    for index, (features,labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        # forward
        # track history if only in train
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
            scores = np.exp(np.max(outputs_test,1))
        if model_type=='knn':
            preds = outputs_test
   
    for i in range(len(labels_test)):
        num_sample_per_class[labels_test[i]] += 1
        if preds[i]==labels_test[i]:
            running_corrects[labels_test[i]] += 1

    acc_per_class = running_corrects / num_sample_per_class
    acc = np.mean(acc_per_class)
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('per-image acc:{:2.4f}; per-class acc:{:2.4f}'.format(running_corrects.sum()/num_sample_per_class.sum(),acc))
    return preds, scores, acc_per_class,acc
    
def loss_fn(recon_xS,recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch):
    criterion = torch.nn.MSELoss(size_average=False)
    mask = yT!=-1
    reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT[mask,:], xT[mask,:])
    cross_reconstruction_loss = criterion(recon_xS2[mask,:], xT[mask,:]) + criterion(recon_xT2[mask,:], xS[mask,:])
    KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT[mask,:] - meanT[mask,:].pow(2) - log_varT[mask,:].exp())
    distance = torch.sqrt(torch.sum((meanS[mask,:] - meanT[mask,:]) ** 2, dim=1) + torch.sum((torch.sqrt(log_varS[mask,:].exp()) - torch.sqrt(log_varT[mask,:].exp())) ** 2, dim=1))
    distance = distance.sum()
    weight = epoch*5e-4
    #print(f'{reconstruction_loss:1.4f}, {cross_reconstruction_loss:1.4f}, {distance:1.4f},{KLD:1.4f}')
    return (reconstruction_loss + cross_reconstruction_loss) / xS.size(0)
    
def train_classifier(classifier, vae, datasets, dataloaders, args, optimizer_cls, scheduler_cls):
    device = args.device
    classifier.train()
    vae.eval()
    acc_per_class = np.zeros((args.num_epochs_cls,datasets['train'].num_class))
    acc = np.zeros((args.num_epochs_cls,))
    for epoch in range(args.num_epochs_cls):
        #print(f'Classifier training epoch {epoch:d}/{args.num_epochs_cls:d}')
        for iteration, (xS,xT,yS,yT) in enumerate(dataloaders['train']):
            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            #x,y = next_batch(vae,batch_size=1024)
            recon_xS,recon_xT = generate_z(xS,xT,vae,device)
            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]            
            recon_xT = recon_xT[mask,:]
            xtrain = torch.cat((xS,xT,recon_xS,recon_xT),dim=0)
            ytrain = torch.cat((yS,yT,yS,yT),dim=0)
            output = classifier(xtrain)
            loss_cls = classifier.lossfunction(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
            # test
        scheduler_cls.step()
        #print(f'epoch:{epoch:02d} ',end='')
        #preds,scores,acc_per_class[epoch,],acc[epoch] = test_model(classifier, datasets['test'], dataloaders['test'],device,model_type='mlp')
    #scipy.io.savemat('./results/'+args.filename+'.mat',mdict={'acc_per_class':acc_per_class,'acc':acc})
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
            recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, d=torch.zeros_like(xS[:,0]).long().to(device))
            recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, d=torch.ones_like(xT[:,0]).long().to(device))
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
    recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, d=torch.zeros_like(xS[:,0]).long().to(device))
    recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, d=torch.ones_like(xT[:,0]).long().to(device))
    return recon_xS2, recon_xT2

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    ts = time.time()
    datasets = {x: TwoModalDataset(phase=x,sourceDomainIndex=args.sourceDomainIndex, targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex) for x in ['train','test']}
    dataloaders={}
    dataloaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers = 8)
    dataloaders['trainall'] = DataLoader(dataset=datasets['train'], batch_size=len(datasets['train']), shuffle=True, num_workers = 8)
    dataloaders['test'] = DataLoader(dataset=datasets['test'], batch_size=len(datasets['test']), shuffle=False, num_workers = 8)
    # define a classifier
    classifier = Classifier(input_dim=2048,num_labels=12).to(device) # train and test a classifier
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=25, gamma=0.1)
    num_epochs_cls = 50
    acc_per_class = np.zeros((args.num_iter,12))
    # define the VAE
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        num_domains = 2,dropout=0.5).to(device)    
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=50, gamma=0.1)

    for iter in range(args.num_iter+5):
        if iter>0:
            # define VAE
            args.encoder_layer_sizes[0] = 2048
            vae = VAE(
                encoder_layer_sizes=args.encoder_layer_sizes,
                latent_size=args.latent_size,
                decoder_layer_sizes=args.decoder_layer_sizes,
                num_domains = 2,dropout=0.5).to(device)    
            optimizer_vae = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
            scheduler_vae = torch.optim.lr_scheduler.StepLR(optimizer_vae, step_size=50, gamma=0.1)
            
            # train VAE
            vae = train_vae(vae, dataloaders['train'], args, optimizer_vae, scheduler_vae)
        # train a classifier
        classifier = Classifier(input_dim=2048,num_labels=12).to(device) # train and test a classifier
        optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
        scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=25, gamma=0.1)
        classifier = train_classifier(classifier, vae, datasets, dataloaders, args, optimizer_cls, scheduler_cls)    
        # classify target samples
        print(f'Iter {iter:02d}: ',end='')
        pseudo_labels, scores, acc_per_class, acc_per_image = test_model(classifier,datasets['test'],dataloaders['test'], device,model_type='mlp')
        # update pseudo-labels,
        datasets['train'].pseudo_label_B = -1*np.ones_like(pseudo_labels)
        #'''
        trustable = np.zeros((len(pseudo_labels),),dtype=np.int32)
        numSelected = np.int32((iter+1)/args.num_iter*len(pseudo_labels)/12)
        for iCls in range(12):
            thisClassFlag = pseudo_labels==iCls
            numThisClass = thisClassFlag.sum()
            
            if numThisClass > 0:
                threshold = sorted(scores[thisClassFlag],reverse=True)[min(numThisClass-1,numSelected)]
                trustable = trustable + np.int32((scores>=threshold) & thisClassFlag)
        datasets['train'].pseudo_label_B[trustable==1] = pseudo_labels[trustable==1]
        print((datasets['train'].pseudo_label_B>-1).sum())
        #'''
        #datasets['train'].pseudo_label_B[scores>0.9-iter*0.1] = pseudo_labels[scores>0.9-iter*0.1]        
        datasets['train'].pseudo_score_B = scores
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs_vae", type=int, default=50)
    parser.add_argument("--num_epochs_cls", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[2048, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 2048])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sourceDomainIndex", type=int, default=1)
    parser.add_argument("--targetDomainIndex", type=int, default=0)
    parser.add_argument("--trialIndex", type=int, default=0)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--num_iter", type=int, default=15)

    args = parser.parse_args()
    
    source = domainSet[args.sourceDomainIndex]
    target = domainSet[args.targetDomainIndex]
    args.filename = 'imageclef-'+source+'-'+target+'-trial'+str(args.trialIndex)+'-numIter-'+str(args.num_iter)+'-vaeEpochs-'+str(args.num_epochs_vae)+'-encoder_layer_sizes'+str(args.encoder_layer_sizes)+'-latSize-'+str(args.latent_size)+'-bs-'+str(args.batch_size)+'lr'+str(args.learning_rate)
    print(args.filename)
    main(args)
