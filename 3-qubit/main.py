#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train the model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import torch
import numpy as np
import os

from configuration import MODEL_CONFIG, TRAIN_CONFIG
from generate_data import Generate_separable_state, Generate_entangled_state
from tools import mkdir, l1_loss, l2_loss_real, l2_loss_complex, roc
from model import Encoder_r, Encoder_f, Generator, Discriminator
from torch.utils.data import Dataset


# from loss import l1_loss,l2_loss

class MyDataset(Dataset):
    def __init__(self, nam, set):
        self.name = nam
        self.set = set
        self.size = len(set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        states = self.set[idx]
        states = np.expand_dims(states, 0)
        states = torch.from_numpy(states).float()
        sample = {'states': states}
        return sample


def train(model, optimizer, device, train_loader, epoch):
    for mod in model:
        mod.train()
    encoder1 = model[0]
    decoder = model[1]
    dis = model[2]
    encoder2 = model[3]
    for batch_idx, data in enumerate(train_loader):
        mix_separable_states = data['states'].to(device)
        mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
        mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
        for opt in optimizer:
            opt.zero_grad()
        [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r, mix_separable_states_i)
        [fake_mix_states_r, fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
        [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r, fake_mix_states_i)
        dis_out_real = dis(mix_separable_states_r, mix_separable_states_i)
        dis_out_fake = dis(fake_mix_states_r, fake_mix_states_i)

        a = 20
        lossadv_g = torch.mean(-dis_out_fake)
        losscon = l1_loss([mix_separable_states_r, mix_separable_states_i], [fake_mix_states_r, fake_mix_states_i])
        lossenc = l2_loss_complex([latent_var1_r, latent_var1_i], [latent_var2_r, latent_var2_i])
        Loss = lossadv_g + a * losscon + 2 * lossenc
        Loss.backward(retain_graph=True)

        for p in dis.parameters():
            p.data.clamp_(-0.01, 0.01)
        optimizer[0].step()
        optimizer[1].step()
        optimizer[3].step()

        for opt in optimizer:
            opt.zero_grad()
        mix_separable_states = data['states'].to(device)
        mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
        mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
        [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r, mix_separable_states_i)
        [fake_mix_states_r, fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
        # [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r, fake_mix_states_i)
        dis_out_real = dis(mix_separable_states_r, mix_separable_states_i)
        dis_out_fake = dis(fake_mix_states_r, fake_mix_states_i)
        lossadv_d = torch.mean(-dis_out_real + dis_out_fake)
        lossadv_d.backward(retain_graph=True)
        for p in dis.parameters():
            p.data.clamp_(-0.01, 0.01)
        optimizer[2].step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data['states']),
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
            mix_separable_states = data['states'].to(device)
            mix_separable_states_r = mix_separable_states[:, :, :, :, 0]
            mix_separable_states_i = mix_separable_states[:, :, :, :, 1]
            [latent_var1_r, latent_var1_i] = encoder1(mix_separable_states_r, mix_separable_states_i)
            [fake_mix_states_r, fake_mix_states_i] = decoder(latent_var1_r, latent_var1_i)
            if flag == 0:
                latent_vector = torch.cat((latent_var1_r, latent_var1_i), dim=1).cpu()
                flag = 1
            [latent_var2_r, latent_var2_i] = encoder2(fake_mix_states_r, fake_mix_states_i)
            loss_enc = l2_loss_complex([latent_var1_r, latent_var1_i], [latent_var2_r, latent_var2_i], test=True).cpu()
            Loss.append(loss_enc)
            # time_end = time.time()
            # time_s = time_s + time_end - time_start
        # print('time cost', time_s, 's')
        return Loss, latent_vector

    # print('Accuracy:%f %%' % (100*correct/total))


if __name__ == '__main__':
    # generate data
    train_size, test_size = TRAIN_CONFIG['train_size'], TRAIN_CONFIG['test_size']
    if TRAIN_CONFIG['generate_data']:
        Generate_separable_state(name='sep_train_set', size=train_size, sub_dim=2, space_number=3,
                                 mix_number=20).generate()
        Generate_separable_state(name='sep_test_set', size=test_size, sub_dim=2, space_number=3,
                                 mix_number=20).generate()
        Generate_entangled_state(name='ent_test_set', size=test_size, sub_dim=2, space_number=3).generate()

    train_data = np.load(TRAIN_CONFIG['train_set_path'])
    test_data_s = np.load(TRAIN_CONFIG['s_test_set_path'])
    test_data_e = np.load(TRAIN_CONFIG['e_test_set_path'])

    train_set = MyDataset(nam='train', set=train_data)
    test_set_s = MyDataset(nam='test_separable', set=test_data_s)
    test_set_e = MyDataset(nam='test_entangled', set=test_data_e)

    batch_size_t = TRAIN_CONFIG['train_config']['batch_size']
    batch_size_v = TRAIN_CONFIG['validation_data_config']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_t, shuffle=True)
    test_loader_s = torch.utils.data.DataLoader(test_set_s, batch_size=batch_size_v, shuffle=True)
    test_loader_e = torch.utils.data.DataLoader(test_set_e, batch_size=batch_size_v, shuffle=True)
    # for batch_idx, data in enumerate(train_loader):
    #     print('in')
    file_writer = open(TRAIN_CONFIG['result_path'], 'a')
    if TRAIN_CONFIG['is_train']:
        AUC = []
        EER = []
        device = TRAIN_CONFIG['train_config']['device']
        c1, k1, c2, k2, c3, k3, d1, d2 = MODEL_CONFIG['c1'], MODEL_CONFIG['k1'], MODEL_CONFIG['c2'], MODEL_CONFIG['k2'], \
                                         MODEL_CONFIG['c3'], MODEL_CONFIG['k3'], MODEL_CONFIG['d1'], MODEL_CONFIG['d2']
        model_1 = Encoder_r(k1=k1, c1=c1, k2=k2, c2=c2, k3=k3, c3=c3, d1=d1, d2=d2).to(device)
        optimizer_1 = torch.optim.RMSprop(model_1.parameters(), lr=0.0001, momentum=0.9)
        model_2 = Generator(k1=k1, c1=c1, k2=k2, c2=c2, k3=k3, c3=c3, d1=d1, d2=d2).to(device)
        optimizer_2 = torch.optim.RMSprop(model_2.parameters(), lr=0.0001, momentum=0.9)
        model_3 = Discriminator(k1=k1, c1=c1, k2=k2, c2=c2, k3=k3, c3=c3, d1=d1, d2=d2).to(device)
        optimizer_3 = torch.optim.RMSprop(model_3.parameters(), lr=0.0001, momentum=0.9)
        model_4 = Encoder_f(k1=k1, c1=c1, k2=k2, c2=c2, k3=k3, c3=c3, d1=d1, d2=d2).to(device)
        optimizer_4 = torch.optim.RMSprop(model_4.parameters(), lr=0.0001, momentum=0.9)
        mkdir('./Model_%d_%d_%d/' % (k1, k2, k3) + 'Channel_%d_%d_%d' % (c1, c2, c3))
        for epoch in range(TRAIN_CONFIG['train_config']['epoch']):
            train([model_1, model_2, model_3, model_4], [optimizer_1, optimizer_2, optimizer_3, optimizer_4], device,
                  train_loader,
                  epoch)
            loss_s, latent_s = test([model_1, model_2, model_3, model_4], test_loader_s, device)
            Loss_s = []
            for i in range(len(loss_s)):
                Loss_s.append(loss_s[i].numpy())
            loss_s = np.reshape(np.array(Loss_s), [len(test_data_s)])
            loss_s = torch.from_numpy(loss_s)
            label_s = torch.zeros(len(loss_s))

            loss_e, latent_e = test([model_1, model_2, model_3, model_4], test_loader_e, device)
            Loss_e = []
            for i in range(len(loss_e)):
                Loss_e.append(loss_e[i].numpy())
            loss_e = np.reshape(np.array(Loss_e), [len(test_data_e)])
            loss_e = torch.from_numpy(loss_e)
            label_e = torch.ones(len(loss_e))

            score = torch.cat((loss_s, loss_e), 0)
            la = torch.cat((label_s, label_e), 0)

            mkdir('./roc/')
            roc_auc, eer = roc(la, score, epoch, saveto='./roc/')
            AUC.append(roc_auc)
            EER.append(eer)
            if roc_auc >= np.max(np.array(AUC)):
                torch.save(model_1, './Model_%d_%d_%d/' % (k1, k2, k3) + 'Channel_%d_%d_%d/' % (c1, c2, c3) + 'model1')
                torch.save(model_2, './Model_%d_%d_%d/' % (k1, k2, k3) + 'Channel_%d_%d_%d/' % (c1, c2, c3) + 'model2')
                torch.save(model_3, './Model_%d_%d_%d/' % (k1, k2, k3) + 'Channel_%d_%d_%d/' % (c1, c2, c3) + 'model3')
                torch.save(model_4, './Model_%d_%d_%d/' % (k1, k2, k3) + 'Channel_%d_%d_%d/' % (c1, c2, c3) + 'model4')
            # test(model, test_loader,device)
        Performance = np.array([AUC, EER])
        Pd_data = pd.DataFrame(Performance.T, columns=["AUC", "EER"])
        file_writer.write('results of %d-%d kernel' % (k1, k2) + str(c1) + '-' + str(c2) + '\n')
        file_writer.write('AUC:' + str(Pd_data.Auc.max()) + '\n')
        file_writer.write('EER:' + str(Pd_data.Err.min()) + '\n')
        file_writer.write('_________________________\n')
        file_writer.flush()
        torch.cuda.empty_cache()


    else:
        device = TRAIN_CONFIG['validation_data_config']['device']
        c1, k1, c2, k2, d1 = MODEL_CONFIG['c1'], MODEL_CONFIG['k1'], MODEL_CONFIG['c2'], MODEL_CONFIG['k2'], \
                             MODEL_CONFIG['d1']
        model_1 = torch.load('./Model_%d_%d/' % (k1, k2) + 'Channel_%d_%d' % (c1, c2) + 'model1')
        model_2 = torch.load('./Model_%d_%d/' % (k1, k2) + 'Channel_%d_%d' % (c1, c2) + 'model2')
        model_3 = torch.load('./Model_%d_%d/' % (k1, k2) + 'Channel_%d_%d' % (c1, c2) + 'model3')
        model_4 = torch.load('./Model_%d_%d/' % (k1, k2) + 'Channel_%d_%d' % (c1, c2) + 'model4')

        loss_s, latent_s = test([model_1, model_2, model_3, model_4], test_loader_s, device)
        Loss_s = []
        for i in range(len(loss_s)):
            Loss_s.append(loss_s[i].numpy())
        loss_s = np.reshape(np.array(Loss_s), [len(test_data_s)])
        loss_s = torch.from_numpy(loss_s)
        label_s = torch.zeros(len(loss_s))

        loss_e, latent_e = test([model_1, model_2, model_3, model_4], test_loader_e, device)
        Loss_e = []
        for i in range(len(loss_e)):
            Loss_e.append(loss_e[i].numpy())
        loss_e = np.reshape(np.array(Loss_e), [len(test_data_e)])
        loss_e = torch.from_numpy(loss_e)
        label_e = torch.ones(len(loss_e))
        score = torch.cat((loss_s, loss_e), 0)
        la = torch.cat((label_s, label_e), 0)

        roc_auc, err = roc(la, score, 101, saveto='./roc/')
