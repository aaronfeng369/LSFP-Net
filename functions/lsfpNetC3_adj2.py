
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from functions.lsp import Project_inf, Wxs, Wtxs
from time import time
dtype = torch.complex64

# define LSFP-Net Block
class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_L = nn.Parameter(torch.tensor([0.0025]))
        self.lambda_S = nn.Parameter(torch.tensor([0.05]))
        self.lambda_spatial_L = nn.Parameter(torch.tensor([5e-2]))
        self.lambda_spatial_S = nn.Parameter(torch.tensor([5e-2]))

        self.gamma = nn.Parameter(torch.tensor([0.5]))
        self.lambda_step = nn.Parameter(torch.tensor([1/10]))

        self.conv1_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3)))
        self.conv2_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv3_forward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))

        self.conv1_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv3_backward_l = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3)))

        self.conv1_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3, 3)))
        self.conv2_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv3_forward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))

        self.conv1_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv2_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3, 3)))
        self.conv3_backward_s = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3, 3)))

    def forward(self, M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S):

        c = self.lambda_step / self.gamma
        nx, ny, nt = M0.size()

        # gradient
        temp_data = torch.reshape(L + S, [nx, ny, nt])
        temp_data = param_E(inv=False, data=temp_data).to(param_d.device)
        gradient = param_E(inv=True, data=temp_data - param_d)
        gradient = torch.reshape(gradient, [nx * ny, nt]).to(param_d.device)

        # pb_L
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

        pb_L = torch.reshape(torch.squeeze(pb_L), [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # y_L
        y_L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # pt_L
        Ut, St, Vt = torch.linalg.svd((c * y_L + pt_L), full_matrices=False)
        temp_St = torch.diag(Project_inf(St, self.lambda_L))
        pt_L = Ut.mm(temp_St).mm(Vt)

        # update p_L
        temp_y_L_input = torch.cat((torch.real(y_L), torch.imag(y_L)), 0).to(torch.float32)
        temp_y_L_input = torch.reshape(temp_y_L_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_L = F.conv3d(temp_y_L_input, self.conv1_forward_l, padding=1)
        temp_y_L = F.relu(temp_y_L)
        temp_y_L = F.conv3d(temp_y_L, self.conv2_forward_l, padding=1)
        temp_y_L = F.relu(temp_y_L)
        temp_y_L_output = F.conv3d(temp_y_L, self.conv3_forward_l, padding=1)

        temp_y_L = temp_y_L_output + p_L
        temp_y_L = temp_y_L[0, :, :, :, :] + 1j * temp_y_L[1, :, :, :, :]
        p_L = Project_inf(c * temp_y_L, self.lambda_spatial_L)

        # new pb_L
        p_L = torch.cat((torch.real(p_L), torch.imag(p_L)), 0).to(torch.float32)
        p_L = torch.reshape(p_L, [2, 32, nx, ny, nt])
        pb_L = F.conv3d(p_L, self.conv1_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L = F.conv3d(pb_L, self.conv2_backward_l, padding=1)
        pb_L = F.relu(pb_L)
        pb_L_output = F.conv3d(pb_L, self.conv3_backward_l, padding=1)

        pb_L = torch.reshape(pb_L_output, [2, nx * ny, nt])
        pb_L = pb_L[0, :, :] + 1j * pb_L[1, :, :]

        # L
        L = L - self.gamma * gradient - self.gamma * pt_L - self.gamma * pb_L

        # adjoint loss: adjloss_L = psi * x * y - psi_t * y * x
        adjloss_L = temp_y_L_output * p_L - pb_L_output * temp_y_L_input

        # pb_S
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

        pb_S = torch.reshape(pb_S, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # y_S
        y_S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        # pt_S
        pt_S = Project_inf(c * Wxs(y_S) + pt_S, self.lambda_S)

        # update p_S
        temp_y_S_input = torch.cat((torch.real(y_S), torch.imag(y_S)), 0).to(torch.float32)
        temp_y_S_input = torch.reshape(temp_y_S_input, [2, nx, ny, nt]).unsqueeze(1)
        temp_y_S = F.conv3d(temp_y_S_input, self.conv1_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S = F.conv3d(temp_y_S, self.conv2_forward_s, padding=1)
        temp_y_S = F.relu(temp_y_S)
        temp_y_S_output = F.conv3d(temp_y_S, self.conv3_forward_s, padding=1)

        temp_y_Sp = temp_y_S_output + p_S
        temp_y_Sp = temp_y_Sp[0, :, :, :, :] + 1j * temp_y_Sp[1, :, :, :, :]
        p_S = Project_inf(c * temp_y_Sp, self.lambda_spatial_S)

        # new pb_S
        p_S = torch.cat((torch.real(p_S), torch.imag(p_S)), 0).to(torch.float32)
        p_S = torch.reshape(p_S, [2, 32, nx, ny, nt])
        pb_S = F.conv3d(p_S, self.conv1_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S = F.conv3d(pb_S, self.conv2_backward_s, padding=1)
        pb_S = F.relu(pb_S)
        pb_S_output = F.conv3d(pb_S, self.conv3_backward_s, padding=1)

        pb_S = torch.reshape(pb_S_output, [2, nx * ny, nt])
        pb_S = pb_S[0, :, :] + 1j * pb_S[1, :, :]

        # S
        S = S - self.gamma * gradient - self.gamma * Wtxs(pt_S) - self.gamma * pb_S

        # adjoint loss: adjloss_S = psi * x * y - psi_t * y * x
        adjloss_S = temp_y_S_output * p_S - pb_S_output * temp_y_S_input

        return [L, S, adjloss_L, adjloss_S, pt_L, pt_S, p_L, p_S]


# define LSFP-Net
class LSFPNet(nn.Module):
    def __init__(self, LayerNo):
        super(LSFPNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for ii in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)


    def forward(self, M0, param_E, param_d):

        M0 = M0[:, :, :, 0] + 1j * M0[:, :, :, 1]
        param_d = param_d[:, :, :, 0] + 1j * param_d[:, :, :, 1]

        nx, ny, nt = M0.size()
        L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_L = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        pt_S = torch.zeros([nx * ny, nt], dtype=dtype).to(param_d.device)
        p_L = torch.zeros([2, 32, nx, ny, nt], dtype=torch.float32).to(param_d.device)
        p_S = torch.zeros([2, 32, nx, ny, nt], dtype=torch.float32).to(param_d.device)

        layers_adj_L = []
        layers_adj_S = []

        for ii in range(self.LayerNo):
            [L, S, layer_adj_L, layer_adj_S, pt_L, pt_S, p_L, p_S] = self.fcs[ii](M0, param_E, param_d, L, S, pt_L, pt_S, p_L, p_S)
            layers_adj_L.append(layer_adj_L)
            layers_adj_S.append(layer_adj_S)

        L = torch.reshape(L, [nx, ny, nt])
        S = torch.reshape(S, [nx, ny, nt])

        return [L, S, layers_adj_L, layers_adj_S]