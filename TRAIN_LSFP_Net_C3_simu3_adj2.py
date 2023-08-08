
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import h5py
from time import time
import torchkbnufft as tkbn
from torch.utils.data import TensorDataset, DataLoader
from functions.utils import trajGR, normabs, MCNUFFT
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from functions.lsfpNetC3_adj2 import LSFPNet
conv_num = 3
dtype = torch.complex64

################# hyper parameters #################
parser = ArgumentParser(description='LSFP-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=100, help='epoch number of end training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--layer_num', type=int, default=3, help='phase number of LSFP-Net')
parser.add_argument('--spf', type=int, default=10, help='spokes per frame from {10,20,50,402}')
parser.add_argument('--fpg', type=int, default=5, help='frames per group from {5, 8, 10}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='results', help='result directory')
parser.add_argument('--train_name', type=str, default='simu_intervention', help='name of train set')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
spf = args.spf
fpg = args.fpg
gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################# Load training data ##############

filename = "./Datasets/IMRI_Simu6_test.mat"
iMRI_data = h5py.File(filename, 'r')
brainI_ref = np.transpose(iMRI_data['brainI_ref_test'])  # reference
c = np.transpose(iMRI_data['b1_test'])   # coil sensitivity maps

brainI_ref = np.transpose(brainI_ref, [3, 0, 1, 2])
c = np.transpose(c, [3, 2, 0, 1])
c.dtype = 'complex128'

overSample = 2
Nitem = brainI_ref.shape[0]
Nsample = brainI_ref.shape[1] * overSample
Nt = brainI_ref.shape[3]
Ncoil = c.shape[1]

Nspokes = spf  # number of spokes per frame
Ng = fpg  # frames per group
Mg = Nt // Ng  # total number of groups
indG = np.arange(0, Nt, 1, dtype=int)
indG = np.reshape(indG, [Mg, Ng])

# prepare training dataset
brainI_ref = torch.from_numpy(brainI_ref)
c = torch.tensor(c, dtype=dtype)

train_dataset = TensorDataset(brainI_ref, c)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

################### prepare NUFFT ################
def prep_nufft(Nsample, Nspokes, Ng):

    overSmaple = 2
    im_size = (int(Nsample/overSmaple), int(Nsample/overSmaple))
    grid_size = (Nsample, Nsample)

    ktraj = trajGR(Nsample, Nspokes * Ng)
    ktraj = torch.tensor(ktraj, dtype=torch.float)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)
    dcomp = dcomp.squeeze()

    ktraju = np.zeros([2, Nspokes * Nsample, Ng], dtype=float)
    dcompu = np.zeros([Nspokes * Nsample, Ng], dtype=complex)

    for ii in range(0, Ng):
        ktraju[:, :, ii] = ktraj[:, (ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]
        dcompu[:, ii] = dcomp[(ii * Nspokes * Nsample):((ii + 1) * Nspokes * Nsample)]

    ktraju = torch.tensor(ktraju, dtype=torch.float)
    dcompu = torch.tensor(dcompu, dtype=dtype)

    nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)  # forward nufft
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)  # backward nufft

    return ktraju, dcompu, nufft_ob, adjnufft_ob

def radial_down_sampling(images, param_E):

    k_und = param_E(inv=False, data=images)
    k_und = k_und.to(images.device)
    im_und = param_E(inv=True, data=k_und)

    return k_und, im_und


ktraj, dcomp, nufft_ob, adjnufft_ob = prep_nufft(Nsample, Nspokes, Ng)
ktraj = ktraj.to(device)
dcomp = dcomp.to(device)
nufft_ob = nufft_ob.to(device)
adjnufft_ob = adjnufft_ob.to(device)

########### Network and Data preparation #############

# new network
model = LSFPNet(layer_num)
model = nn.DataParallel(model, [0])
model = model.to(device)

# print parameter number
print_flag = 0
if print_flag:
    print(model)
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
Mloss = nn.MSELoss()

# dir of model and log
model_dir = "./%s/LSFP_Net_C%d_B%d_SPF%d_FPG%d_adj2" % (args.model_dir, conv_num, layer_num, spf, fpg)
log_name = "./%s/Log_LSFP_Net_C%d_B%d_SPF%d_FPG%d_adj2.txt" % (args.log_dir, conv_num, layer_num, spf, fpg)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

################### training loop #####################
if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

for epoch_i in range(start_epoch + 1, end_epoch + 1):


    for idx, (x_train, c_train) in enumerate(train_loader):

        random_group = torch.randperm(Mg)
        x_train_it = x_train.squeeze()
        smap1 = c_train.squeeze().to(device)
        param_E = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp, smap1)

        for p in random_group[0:1]:
            ground_truth = x_train_it[:, :, indG[p, :]]
            gt = ground_truth.type(dtype).to(device)
            param_d, M0 = radial_down_sampling(gt, param_E)

            time1 = time()
            [L, S, loss_layers_adj_L, loss_layers_adj_S] = model(M0, param_E, param_d)
            M = torch.abs(L + S)
            M_recon = M.cpu().data.numpy()
            time2 = time()

            temp_ground_truth = ground_truth.type(torch.float32).to(device)
            loss_constraint_L = torch.square(torch.mean(loss_layers_adj_L[0])) / layer_num
            loss_constraint_S = torch.square(torch.mean(loss_layers_adj_S[0])) / layer_num

            for k in range(layer_num - 1):
                loss_constraint_S += torch.square(torch.mean(loss_layers_adj_S[k + 1])) / layer_num
                loss_constraint_L += torch.square(torch.mean(loss_layers_adj_L[k + 1])) / layer_num

            gamma = torch.Tensor([0.01]).to(device)
            loss_ref = Mloss(M, temp_ground_truth)
            loss = loss_ref + torch.mul(gamma, loss_constraint_L + loss_constraint_S)

            # plt.figure()
            # plt.subplot(1, 3, 1)
            # plt.imshow(abs(ground_truth[:, :, Ng-1]), 'gray')
            # plt.subplot(1, 3, 2)
            # plt.imshow(abs(M0.numpy()[:, :, Ng-1]), 'gray')
            # plt.subplot(1, 3, 3)
            # plt.imshow(abs(M_recon[:, :, Ng-1]), 'gray')
            # plt.show()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_data = "[%02d/%02d] Loss_ref: %.5f, Loss_adj_L: %.5f, Loss_adj_S: %.5f, Loss: %.5f\n" % \
                          (epoch_i, end_epoch, loss_ref, loss_constraint_L, loss_constraint_S, loss)
            output_data2 = "[%02d/%02d] Loss_ref: %.5f, Loss_adj_L: %.5f, Loss_adj_S: %.5f,Loss: %.5f，time: %.4f\n" % \
                           (epoch_i, end_epoch, loss_ref, loss_constraint_L, loss_constraint_S, loss, (time2-time1))
            print(output_data2)

    output_file = open(log_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
