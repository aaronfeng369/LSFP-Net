
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import imageio
from time import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import platform
from argparse import ArgumentParser

import torchkbnufft as tkbn
from functions.utils import normabs, crop, MCNUFFT, trajGR
from functions.lsfpNetC3_adj2 import LSFPNet
conv_num = 3
dtype = torch.complex64

################# hyper parameters ###########################
parser = ArgumentParser(description='LSFP-Net')

parser.add_argument('--epoch_num', type=int, default=100, help='epoch number of model')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--layer_num', type=int, default=3, help='phase number of LSFP-Net')
parser.add_argument('--spf', type=int, default=10, help='spokes per frame from {10,20,50,402}')
parser.add_argument('--fpg', type=int, default=5, help='frames per group from {5, 8, 10}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='simu_results', help='result directory')
parser.add_argument('--train_name', type=str, default='Datasets', help='name of train set')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
spf = args.spf
fpg = args.fpg
gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################# Load training data ##############

# load data
filename = "./Datasets/IMRI_Simu6_test.mat"
iMRI_data = h5py.File(filename, 'r')
brainI_ref = np.transpose(iMRI_data['brainI_ref_test'])  # reference
c = np.transpose(iMRI_data['b1_test'])   # coil sensitivity maps

brainI_ref = np.transpose(brainI_ref, [3, 0, 1, 2])
c = np.transpose(c, [3, 2, 0, 1])
c.dtype = 'complex128'

# parameters
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

test_dataset = TensorDataset(brainI_ref, c)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

########### Network preparation #############

# new network
model = LSFPNet(layer_num)
model = nn.DataParallel(model, [0])
model = model.to(device)

####################### load parameters of trained model ##################
model_dir = "./%s/LSFP_Net_C%d_B%d_SPF%d_FPG%d_adj2" % (args.model_dir, conv_num, layer_num, spf, fpg)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num), map_location='cuda:0'))

# print parameter number
print_flag = 1
if print_flag:
    print(model)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total:%d, Trainable:%d' % (total_num,trainable_num))

# folder for saving results
output_file_name = "./%s/Simu_Results_LSFP_Net_C%d_B%d_SPF%d_FPG%d_Epoch%d_adj2.txt" % \
                   (args.result_dir, conv_num, layer_num, Nspokes, Ng, epoch_num)
matName = "./%s/Simu_Results_LSFP_Net_C%d_B%d_SPF%d_FPG%d_Epoch%d_adj2.mat" % \
          (args.result_dir, conv_num, layer_num, Nspokes, Ng, epoch_num)

result_dir = os.path.join(args.result_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

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


##################### Test network ####################
Ny = int(Nsample / overSample)
Nx = int(Nsample / overSample)
mC = 0

Init_All = np.zeros([Ny, Nx, Nitem], dtype=complex)
Rec_All = np.zeros([Ny, Nx, Nitem], dtype=complex)

Init_Time_All = np.zeros([1, Nitem], dtype=np.float32)
Init_PSNR_All = np.zeros([1, Nitem], dtype=np.float32)
Init_SSIM_All = np.zeros([1, Nitem], dtype=np.float32)

Rec_Time_All = np.zeros([1, Nitem], dtype=np.float32)
Rec_PSNR_All = np.zeros([1, Nitem], dtype=np.float32)
Rec_SSIM_All = np.zeros([1, Nitem], dtype=np.float32)

print('\n')
print("LSFP-Net Reconstruction Start")

with torch.no_grad():
    for idx in range(0, Nitem):

        x_test_it = brainI_ref[idx, :, :, :]
        smap1 = c[idx, :, :, :]  # coil sensitivity maps
        smap1 = torch.tensor(smap1, dtype=dtype)  # convert smaps to tensors
        smap1 = smap1.to(device)

        param_E = MCNUFFT(nufft_ob, adjnufft_ob, ktraj, dcomp, smap1)

        for p in range(Mg-1, Mg):

            gnd_it = x_test_it[:, :, indG[p, :]]
            gnd = gnd_it.type(dtype).to(device)

            torch.cuda.synchronize()
            start1 = time()
            param_d, M0 = radial_down_sampling(gnd, param_E)
            torch.cuda.synchronize()
            end1 = time()

            torch.cuda.synchronize()
            start2 = time()
            [L, S, loss_L, loss_S] = model(M0, param_E, param_d)
            M = L + S
            M_recon = np.squeeze(M.cpu().data.numpy())
            torch.cuda.synchronize()
            end2 = time()

            X_init = crop(M0.numpy(), int(Nsample / overSample) - mC, int(Nsample / overSample) - mC)
            X_rec = crop(M_recon, int(Nsample / overSample) - mC, int(Nsample / overSample) - mC)
            t1 = end1 - start1
            t2 = end2 - start2

            Init_All[:, :, idx] = X_init[:, :, Ng - 1]
            Rec_All[:, :, idx] = X_rec[:, :, Ng - 1]

            Init_Time_All[:, idx] = t1
            Rec_Time_All[:, idx] = t2

            # Error map
            ground_truth = gnd_it.numpy()
            # init_error = abs(abs(ground_truth[0:128, :, Ng - 1]) - abs(X_init[0:128, :, Ng - 1]))
            # rec_error = abs(abs(ground_truth[0:128, :, Ng - 1]) - abs(X_rec[0:128, :, Ng - 1]))

            init_error = abs(normabs(ground_truth[0:128, :, Ng - 1]) - normabs(X_init[0:128, :, Ng - 1]))
            rec_error = abs(normabs(ground_truth[0:128, :, Ng - 1]) - normabs(X_rec[0:128, :, Ng - 1]))

            # PSNR and SSIM
            Init_PSNR_All[:, idx] = psnr(normabs(X_init[0:128, :, Ng - 1]) * 255.,
                                         normabs(ground_truth[0:128, :, Ng - 1]) * 255.,
                                         data_range=255)
            Init_SSIM_All[:, idx] = ssim(normabs(X_init[0:128, :, Ng - 1]) * 255.,
                                         normabs(ground_truth[0:128, :, Ng - 1]) * 255.,
                                         data_range=255)
            Rec_PSNR_All[:, idx] = psnr(normabs(X_rec[0:128, :, Ng - 1]) * 255., normabs(ground_truth[0:128, :, Ng - 1]) * 255.,
                                        data_range=255)
            Rec_SSIM_All[:, idx] = ssim(normabs(X_rec[0:128, :, Ng - 1]) * 255., normabs(ground_truth[0:128, :, Ng - 1]) * 255.,
                                        data_range=255)

            print("[%02d/%02d] Time is %.2fs, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (
                p+1, Mg, t1, Init_PSNR_All[:, idx], Init_SSIM_All[:, idx]))
            print("[%02d/%02d] Time is %.2fs, Proposed PSNR is %.2f, Proposed SSIM is %.4f\n" % (
                p+1, Mg, t2, Rec_PSNR_All[:, idx], Rec_SSIM_All[:, idx]))

            rec_results = np.append(normabs(ground_truth[0:128, :, Ng - 1]), normabs(X_init[0:128, :, Ng - 1]), axis=1)
            rec_results = np.append(rec_results, normabs(X_rec[0:128, :, Ng - 1]), axis=1)

            error_maps = np.append(np.zeros_like(ground_truth[0:128, :, Ng - 1]), np.zeros_like(ground_truth[0:128, :, Ng - 1]), axis=1)
            error_maps = np.append(error_maps, rec_error, axis=1)

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(rec_results, 'gray')
            plt.axis('off')
            plt.colorbar()
            # plt.title('Reference, NUFFT, LSFP-Net')

            plt.subplot(2, 1, 2)
            plt.imshow(error_maps, cmap=plt.cm.viridis)
            plt.axis('off')
            plt.colorbar()
            # plt.title('Error maps')

            plt.tight_layout()
            plt.show()

# save PSNR and SSIM info
init_data =   "SPF is %d, FPG is %d, Avg Initial Time/PSNR/SSIM is %.2f/%.2f/%.4f\n" \
              % (spf, fpg, np.mean(Init_Time_All), np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "SPF is %d, FPG is %d, Avg Proposed Time/PSNR/SSIM is %.2f/%.2f/%.4f, Std is %.2f/%.2f/%.4f\n" \
              % (spf, fpg, np.mean(Rec_Time_All), np.mean(Rec_PSNR_All), np.mean(Rec_SSIM_All),
                 np.std(Rec_Time_All), np.std(Rec_PSNR_All), np.std(Rec_SSIM_All))
print(init_data)
print(output_data)

output_file = open(output_file_name, 'a')
output_file.write(init_data)
output_file.write(output_data)
output_file.close()

# save mat
sio.savemat(matName, {'Init_All': Init_All, 'Rec_All': Rec_All, 'Init_Time_All': Init_Time_All, 'Rec_Time_All': Rec_Time_All,
                      'Init_PSNR_All': Init_PSNR_All, 'Init_SSIM_All': Init_SSIM_All, 'Rec_PSNR_All': Rec_PSNR_All, 'Rec_SSIM_All': Rec_SSIM_All})
print('\n')
print('LSFP-Net Reconstruction end.')
