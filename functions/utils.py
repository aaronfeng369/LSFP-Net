# General functions for multi-coil radial sampling MRI
# Apr-12-2021==Zhao He==Original code
# May-11-2021==Zhao He==Add MCNUFFT function
# May-16-2021==Zhao He==Add crop function for 3-dimension data
# May-17-2021==Zhao He==Revise MCNUFFT function
# Jun-20-2021==Zhao He==Add comments

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import torchkbnufft as tkbn
from time import time
# import matplotlib.pyplot as plt

dtype = torch.complex64


def fft2_mri(x):
    # return torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(x)))
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))


def ifft2_mri(x):
    # return torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(x)))
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))


def crop(imageIn, Nouty, Noutx):
    '''
    function for cropping image
    :param imageIn: image to be cropped
    :param Nouty: image to be cropped
    :param Noutx: image to be cropped
    :return: imageOut: image after cropped
    '''
    if len(imageIn.shape) == 4:  # if imageIn is 4-dimension data

        Ny = imageIn.shape[0]
        Nx = imageIn.shape[1]
        cy = int(Nouty / 2)
        cx = int(Noutx / 2)
        imageOut = imageIn[int(Ny / 2 - cy):int(Ny / 2 + cy), int(Nx / 2 - cx):int(Nx / 2 + cx), :, :]

    elif len(imageIn.shape) == 3:  # if 3-dimension data

        Ny = imageIn.shape[0]
        Nx = imageIn.shape[1]
        cy = int(Nouty / 2)
        cx = int(Noutx / 2)
        imageOut = imageIn[int(Ny / 2 - cy):int(Ny / 2 + cy), int(Nx / 2 - cx):int(Nx / 2 + cx), :]

    else:  # if imageIn is 2-dimension image

        Ny = imageIn.shape[0]
        Nx = imageIn.shape[1]
        cy = int(Nouty / 2)
        cx = int(Noutx / 2)
        imageOut = imageIn[int(Ny / 2 - cy):int(Ny / 2 + cy), int(Nx / 2 - cx):int(Nx / 2 + cx)]

    return imageOut


def normabs(data):
    '''
    function for image normalization
    :param data: image to be normalized
    :return: image after normalized
    '''
    data = np.abs(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def trajGR(Nkx, Nspokes):
    '''
    function for generating golden-angle radial sampling trajectory
    :param Nkx: spoke length
    :param Nspokes: number of spokes
    :return: ktraj: golden-angle radial sampling trajectory
    '''
    # ga = np.deg2rad(180 / ((np.sqrt(5) + 1) / 2))
    ga = np.pi * ((1 - np.sqrt(5)) / 2)
    kx = np.zeros(shape=(Nkx, Nspokes))
    ky = np.zeros(shape=(Nkx, Nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, Nkx)
    for i in range(1, Nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    return ktraj

class MCTOEP(nn.Module):
    def __init__(self, toep_ob, smaps):
        super(MCTOEP, self).__init__()
        self.toep_ob = toep_ob
        # self.toep_kernal_all = toep_kernal_all
        self.smaps = smaps.unsqueeze(0)

    def forward(self, data, toep_kernal_all):

        data = torch.squeeze(data)  # delete redundant dimension
        x = torch.zeros(data.shape, dtype=dtype)

        if len(data.shape) > 2:  # multi-frame

            for ii in range(0, data.shape[-1]):

                image = data[:, :, ii]
                image = image.unsqueeze(0).unsqueeze(0)
                image = image.to(self.smaps.device)

                toep_kernal = toep_kernal_all[:, :, ii, :]
                toep_kernal = toep_kernal[:, :, 0] + 1j * toep_kernal[:, :, 1]
                x_temp = self.toep_ob(image, toep_kernal, smaps=self.smaps)

                x[:, :, ii] = torch.squeeze(x_temp)

        else:  # single frame

            toep_kernal = tkbn.calc_toeplitz_kernel(self.ktraj, self.im_size, weights=self.dcomp)
            toep_kernal = toep_kernal.to(data.device)
            x = self.toep_ob(data, toep_kernal, self.smaps)

        return x


class MCNUFFT(nn.Module):
    def __init__(self, nufft_ob, adjnufft_ob, ktraj, dcomp, smaps):
        super(MCNUFFT, self).__init__()
        self.nufft_ob = nufft_ob
        self.adjnufft_ob = adjnufft_ob
        self.ktraj = torch.squeeze(ktraj)
        self.dcomp = torch.squeeze(dcomp)
        self.smaps = smaps.unsqueeze(0)

    def forward(self, inv, data):
        data = torch.squeeze(data)  # delete redundant dimension
        Nx = self.smaps.shape[2]
        Ny = self.smaps.shape[3]

        if inv:  # adjoint nufft
            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([Nx, Ny, data.shape[2]], dtype=dtype)

                for ii in range(0, data.shape[2]):
                    kd = data[:, :, ii]
                    k = self.ktraj[:, :, ii]
                    d = self.dcomp[:, ii]

                    kd = kd.unsqueeze(0)
                    d = d.unsqueeze(0).unsqueeze(0)

                    tt1 = time()
                    x_temp = self.adjnufft_ob(kd * d, k, smaps=self.smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)
                    tt2 = time()
                    # print('adjnufft time is %.6f' % (tt2 - tt1))

                    # plt.figure()
                    # plt.imshow(np.abs(x_temp.numpy()), 'gray')
                    # plt.show()

            else:  # single frame

                kd = data.unsqueeze(0)
                d = self.dcomp.unsqueeze(0).unsqueeze(0)
                x = self.adjnufft_ob(kd * d, self.ktraj, smaps=self.smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        else:  # forward nufft
            if len(data.shape) > 2:  # multi-frame

                x = torch.zeros([self.smaps.shape[1], self.ktraj.shape[1], data.shape[-1]], dtype=dtype)

                for ii in range(0, data.shape[-1]):
                    image = data[:, :, ii]
                    k = self.ktraj[:, :, ii]

                    image = image.unsqueeze(0).unsqueeze(0)
                    x_temp = self.nufft_ob(image, k, smaps=self.smaps)
                    x[:, :, ii] = torch.squeeze(x_temp) / np.sqrt(Nx * Ny)

            else:  # single frame

                image = data.unsqueeze(0).unsqueeze(0)
                x = self.nufft_ob(image, self.ktraj, smaps=self.smaps)
                x = torch.squeeze(x) / np.sqrt(Nx * Ny)

        return x
