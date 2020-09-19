import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear,ComplexConvTranspose2d
from complexFunctions import complex_relu, complex_max_pool2d
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
# from loss import l1_loss,l2_loss

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Folder have been made!')

def l1_loss(input, target, test = False):
    """ L1 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L1 distance between input and output
    """
    if test:
        return torch.mean(torch.abs(input[0] - target[0]) + torch.abs(input[1] - target[1]),dim = 1, keepdim=False)
    else:
        return torch.mean(torch.abs(input[0] - target[0]) + torch.abs(input[1] - target[1]))

##
def l2_loss_real(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def l2_loss_complex(input, target, test = False):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if test:
        return torch.mean(torch.pow((input[0]-target[0]), 2) + torch.pow((input[1]-target[1]), 2),dim = 1, keepdim=False)
    else:
        return torch.mean(torch.pow((input[0]-target[0]), 2) + torch.pow((input[1]-target[1]), 2))



train_set = np.load('separable_train_set.npy')
test_set = np.load('separable_test_set.npy')
test_set_f = np.load('entangled_set.npy')
test_set_ba = np.load('entangled_set.npy')
test_set_bb = np.load('entangled_set.npy')
test_set_bc = np.load('entangled_set.npy')

class MyDataset(Dataset):
    def __init__(self, nam):
        self.name = nam
        if self.name =='train':
            self.size = len(train_set)
        else:
            self.size = len(test_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.name == 'train':
            image = train_set[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        elif self.name == 'test':
            image = test_set[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        elif self.name == 'f_entangled':
            image = test_set_f[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        elif self.name == 'b_entangled_a':
            image = test_set_ba[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        elif self.name == 'b_entangled_b':
            image = test_set_bb[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        elif self.name == 'b_entangled_c':
            image = test_set_bc[idx]
            image = np.expand_dims(image, 0)
            image = torch.from_numpy(image).float()
            sample = {'data': image}
        return sample

train_data = MyDataset(nam='train')
test_data = MyDataset(nam='test')
test_data_f = MyDataset(nam='f_entangled')
test_data_b_a = MyDataset(nam='b_entangled_a')
test_data_b_b = MyDataset(nam='b_entangled_b')
test_data_b_c = MyDataset(nam='b_entangled_c')
# for (cnt, i) in enumerate(train_data):
#     image = i['image']
#     label = i['label']
#     print('in')

batch_size = 512
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2000, shuffle=True)
test_loader_f = torch.utils.data.DataLoader(test_data_f, batch_size=2000, shuffle=True)
test_loader_b_a = torch.utils.data.DataLoader(test_data_b_a, batch_size=2000, shuffle=True)
test_loader_b_b = torch.utils.data.DataLoader(test_data_b_b, batch_size=2000, shuffle=True)
test_loader_b_c = torch.utils.data.DataLoader(test_data_b_c, batch_size=2000, shuffle=True)
# for batch_idx, data in enumerate(train_loader):
#     print('in')
c3 = 50
class Encoder(nn.Module):
    def __init__(self,c1 = 20, c2 = 30, c3 = 50, d1 = 96, d2 = 16):
        super(Encoder, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, 3, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, 3, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, 3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * c3)
        xi = xi.view(-1, 2 * 2 * c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        # x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return xr, xi


class Decoder(nn.Module):

    def __init__(self,c1 = 20, c2 = 30, c3 = 50, d1 = 96, d2 = 16):
        super(Decoder, self).__init__()
        self.convt1 = ComplexConvTranspose2d(c1, 1, 3, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.convt2 = ComplexConvTranspose2d(c2, c1, 3, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.convt3 = ComplexConvTranspose2d(c3, c2, 3, 1, padding=0)  # k = 2,p' = k - 1
        self.fc1 = ComplexLinear(d1, 2 * 2 * c3)
        self.fc2 = ComplexLinear(d2, d1)

    def forward(self, xr, xi):
        # imaginary part to zero
        xr, xi = self.fc2(xr, xi)
        xr, xi = complex_relu(xr, xi)

        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)
        xr, xi = self.fc1(xr, xi)
        xr = xr.view(-1, c3, 2, 2)
        xi = xi.view(-1, c3, 2, 2)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt3(xr, xi)
        xr, xi = self.bn2(xr, xi)

        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt2(xr, xi)
        xr, xi = self.bn1(xr, xi)

        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt1(xr, xi)

        return xr, xi

class Encoder2(nn.Module):

    def __init__(self,c1 = 20, c2 = 30, c3 = 50, d1 = 96, d2 = 16):
        super(Encoder2, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, 3, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, 3, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, 3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * c3)
        xi = xi.view(-1, 2 * 2 * c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        # x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return xr, xi


class Dis(nn.Module):

    def __init__(self,c1 = 20, c2 = 30, c3 = 50, d1 = 96, d2 = 16):
        super(Dis, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, 3, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, 3, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, 3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * c3)
        xi = xi.view(-1, 2 * 2 * c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return x


def train(model, optimizer, device, train_loader, epoch):
    for mod in model:
        mod.train()
    encoder1 = model[0]
    decoder = model[1]
    dis = model[2]
    encoder2 = model[3]
    for batch_idx, data in enumerate(train_loader):
        mix_separable_states = data['data'].to(device)
        mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
        mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
        for opt in optimizer:
            opt.zero_grad()
        [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r,mix_separable_states_i)
        [fake_mix_states_r,fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
        [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r,fake_mix_states_i)
        dis_out_real = dis(mix_separable_states_r,mix_separable_states_i)
        dis_out_fake = dis(fake_mix_states_r,fake_mix_states_i)

        a = 20
        lossadv_d = torch.mean(-dis_out_real + dis_out_fake)
        lossadv_g = torch.mean(-dis_out_fake)
        losscon = l1_loss([mix_separable_states_r,mix_separable_states_i],[fake_mix_states_r,fake_mix_states_i])
        lossenc = l2_loss_complex([latent_var1_r, latent_var1_i],[latent_var2_r, latent_var2_i])
        Loss = lossadv_g + a * losscon + 2 * lossenc
        Loss.backward(retain_graph=True)

        for p in dis.parameters():
            p.data.clamp_(-0.01, 0.01)
        optimizer[0].step()
        optimizer[1].step()
        optimizer[3].step()

        for opt in optimizer:
            opt.zero_grad()
        mix_separable_states = data['data'].to(device)
        mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
        mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
        [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r, mix_separable_states_i)
        [fake_mix_states_r, fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
        [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r, fake_mix_states_i)
        dis_out_real = dis(mix_separable_states_r, mix_separable_states_i)
        dis_out_fake = dis(fake_mix_states_r, fake_mix_states_i)
        lossadv_d = 1 *torch.mean(-dis_out_real + dis_out_fake)
        lossadv_d.backward(retain_graph=True)
        for p in dis.parameters():
            p.data.clamp_(-0.01, 0.01)
        optimizer[2].step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data['data']),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                Loss.item())
            )

def test(model, testloader, device):
    for mod in model:
        mod.eval()
    correct = 0.
    total = 0.
    encoder1 = model[0]
    decoder = model[1]
    dis = model[2]
    encoder2 = model[3]

    with torch.no_grad():
        Loss = []
        # time_s = 0
        flag = 0
        for data in testloader:
            # import time
            # time_start = time.time()
            mix_separable_states = data['data'].to(device)
            mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
            mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
            [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r, mix_separable_states_i)
            [fake_mix_states_r, fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
            if flag == 0:
                latent_vector = torch.cat((latent_var1_r,latent_var1_i),dim = 1).cpu()
                flag = 1
            [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r, fake_mix_states_i)
            dis_out_real = dis(mix_separable_states_r, mix_separable_states_i)
            dis_out_fake = dis(fake_mix_states_r, fake_mix_states_i)
            lossadv = l2_loss_real(dis_out_real, dis_out_fake)
            losscon = l1_loss([mix_separable_states_r, mix_separable_states_i], [fake_mix_states_r, fake_mix_states_i])
            lossenc = l2_loss_complex([latent_var1_r, latent_var1_i], [latent_var2_r, latent_var2_i],test = True).cpu()
            Loss.append(lossenc)
            # time_end = time.time()
            # time_s = time_s + time_end - time_start
        # print('time cost', time_s, 's')
        return lossenc,latent_vector

def roc(labels, scores, number, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print(1-eer)
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))

        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, 'ROC_%d.png'%(number)))
        plt.close()

    return roc_auc,eer


    # print('Accuracy:%f %%' % (100*correct/total))
if __name__ == '__main__':
# Run training on 50 epochs
    T = 'test'
    file_writer = open('result.txt', 'a')
    if T == 'train':
        for c1 in [10]:
            c2 = 30
            for d1 in [96]:
                AUC_f = []
                ERR_f = []
                AUC_ba = []
                ERR_ba = []
                AUC_bb = []
                ERR_bb = []
                AUC_bc = []
                ERR_bc = []
                device = torch.device("cuda:0")
                model1 = Encoder(c1=c1,c2=c2,d1=d1,d2=10).to(device)
                optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=0.0001, momentum=0.9)
                model2 = Decoder(c1=c1,c2=c2,d1=d1,d2 = 10).to(device)
                optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=0.0001, momentum=0.9)
                model3 = Dis(c1=c1,c2=c2,d1=d1,d2=10).to(device)
                optimizer3 = torch.optim.RMSprop(model3.parameters(), lr=0.0001, momentum=0.9)
                model4 = Encoder2(c1=c1,c2=c2,d1=d1,d2=10).to(device)
                optimizer4 = torch.optim.RMSprop(model4.parameters(), lr=0.0001, momentum=0.9)
                mkdir('./Model/Model_%d_%d'%(c1,c2))
                for epoch in range(70):
                    train([model1, model2, model3, model4], [optimizer1, optimizer2, optimizer3, optimizer4], device, train_loader,
                          epoch)
                    loss,latent1 = test([model1, model2, model3, model4], test_loader, device)
                    label = torch.zeros(len(loss))
                    loss_f,latent_f = test([model1, model2, model3, model4], test_loader_f, device)
                    label_f = torch.ones(len(loss_f))
                    loss_ba,latent_ba = test([model1, model2, model3, model4], test_loader_b_a, device)
                    label_ba = torch.ones(len(loss_ba))
                    loss_bb,latent_bb = test([model1, model2, model3, model4], test_loader_b_b, device)
                    label_bb = torch.ones(len(loss_bb))
                    loss_bc,latent_bc = test([model1, model2, model3, model4], test_loader_b_c, device)
                    label_bc = torch.ones(len(loss_bc))


                    score = torch.cat((loss, loss_f), 0)
                    la = torch.cat((label, label_f), 0)
                    mkdir('./roc_f/')
                    roc_auc_f,err_f = roc(la, score, epoch, saveto='./roc_f/')
                    AUC_f.append(roc_auc_f)
                    ERR_f.append(err_f)

                    score = torch.cat((loss, loss_ba), 0)
                    la = torch.cat((label, label_ba), 0)
                    mkdir('./roc_ba/')
                    roc_auc_ba,err_ba = roc(la, score, epoch, saveto='./roc_ba/')
                    AUC_ba.append(roc_auc_ba)
                    ERR_ba.append(err_ba)

                    score = torch.cat((loss, loss_bb), 0)
                    la = torch.cat((label, label_bb), 0)
                    mkdir('./roc_bb/')
                    roc_auc_bb,err_bb = roc(la, score, epoch, saveto='./roc_bb/')
                    AUC_bb.append(roc_auc_bb)
                    ERR_bb.append(err_bb)

                    score = torch.cat((loss, loss_bc), 0)
                    la = torch.cat((label, label_bc), 0)
                    mkdir('./roc_bc/')
                    roc_auc_bc,err_bc = roc(la, score, epoch, saveto='./roc_bc/')
                    AUC_bc.append(roc_auc_bc)
                    ERR_bc.append(err_bc)

                    if err_f < np.max(np.array(ERR_f)):
                        torch.save(model1, './Model/Model_%d_%d/model1'%(c1,c2))
                        torch.save(model2, './Model/Model_%d_%d/model2'%(c1,c2))
                        torch.save(model3, './Model/Model_%d_%d/model3'%(c1,c2))
                        torch.save(model4, './Model/Model_%d_%d/model4'%(c1,c2))
                    # test(model, test_loader,device)
                Acc_f = np.array([AUC_f, ERR_f])
                Pd_data = pd.DataFrame(Acc_f.T, columns=["Auc", "Err"])
                file_writer.write('f_result' + str(c1) + '-' + str(c2) + '\n')
                file_writer.write('Auc:' + str(Pd_data.Auc.max()) + '\n')
                file_writer.write('Er:' + str(Pd_data.Err.min()) + '\n')
                file_writer.write('_________________________\n')

                Acc_ba = np.array([AUC_ba, ERR_ba])
                Pd_data = pd.DataFrame(Acc_ba.T, columns=["Auc", "Err"])
                file_writer.write('f_result' + str(c1) + '-' + str(c2) + '\n')
                file_writer.write('Auc:' + str(Pd_data.Auc.max()) + '\n')
                file_writer.write('Er:' + str(Pd_data.Err.min()) + '\n')
                file_writer.write('_________________________\n')

                Acc_bb = np.array([AUC_bb, ERR_bb])
                Pd_data = pd.DataFrame(Acc_bb.T, columns=["Auc", "Err"])
                file_writer.write('f_result' + str(c1) + '-' + str(c2) + '\n')
                file_writer.write('Auc:' + str(Pd_data.Auc.max()) + '\n')
                file_writer.write('Er:' + str(Pd_data.Err.min()) + '\n')
                file_writer.write('_________________________\n')

                Acc_bc = np.array([AUC_bc, ERR_bc])
                Pd_data = pd.DataFrame(Acc_bc.T, columns=["Auc", "Err"])
                file_writer.write('f_result' + str(c1) + '-' + str(c2) + '\n')
                file_writer.write('Auc:' + str(Pd_data.Auc.max()) + '\n')
                file_writer.write('Er:' + str(Pd_data.Err.min()) + '\n')
                file_writer.write('_________________________\n')


                Acc = np.array([AUC_bc, ERR_bc])
                Pd_data = pd.DataFrame(Acc.T, columns=["Auc", "Err"])
                file_writer.write('results of 3-1 kernel' + str(c1) + '-' + str(c2) + '\n')
                file_writer.write('Auc:' + str(Pd_data.Auc.max()) + '\n')
                file_writer.write('Er:' + str(Pd_data.Err.min()) + '\n')
                file_writer.write('_________________________\n')
                file_writer.flush()
                torch.cuda.empty_cache()


    if T == 'test':
        model_test_1 = torch.load('./Model/Model_10_30/model1')
        model_test_2 = torch.load('./Model/Model_10_30/model2')
        model_test_3 = torch.load('./Model/Model_10_30/model3')
        model_test_4 = torch.load('./Model/Model_10_30/model4')
        loss = test([model_test_1, model_test_2, model_test_3, model_test_4], test_loader, device).cpu()
        label = torch.zeros(len(loss))
        loss1 = test([model_test_1, model_test_2, model_test_3, model_test_4], test_loader1, device).cpu()
        label1 = torch.ones(len(loss1))
        import matplotlib.pyplot as plt
        params = dict(histtype='stepfilled', alpha=0.3, normed=False, bins=50)
        plt.hist(loss1, **params, label='Entangled States')
        plt.hist(loss, **params, label='Separable States')
        plt.xlabel("Score")
        plt.ylabel("frequency")
        plt.legend()
        plt.savefig('./hist.png')
        score = torch.cat((loss,loss1),0)
        la = torch.cat((label, label1), 0)

        roc_auc = roc(la,score,saveto='./')