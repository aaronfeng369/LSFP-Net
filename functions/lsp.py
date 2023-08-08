
import torch
import numpy as np
from time import time

dtype=torch.complex64

def Project_inf(x, c):
    # torch.cuda.synchronize()
    # start1 = time()
    # x_max = torch.maximum((abs(x)/c), torch.ones(x.shape).to(x.device))
    x_max = torch.maximum((abs(x) / c), torch.tensor(1))
    x_max = x_max.to(dtype)
    s = torch.div(x, x_max)

    # torch.cuda.synchronize()
    # end1 = time()
    # print('soft is %.4f' % (end1 - start1))

    return s

def Wxs(x):

    temp_x = torch.zeros_like(x, dtype=dtype)
    temp_x[:, 0:x.shape[1]-2] = x[:, 1:x.shape[1]-1]
    temp_x[:, x.shape[1]-1] = temp_x[:, x.shape[1]-1]
    res = temp_x - x

    return res

def Wtxs(x):

    temp_x = torch.zeros_like(x, dtype=dtype)
    temp_x[:, 0] = temp_x[:, 0]
    temp_x[:, 1:x.shape[1]-1] = x[:, 0:x.shape[1]-2]
    res = temp_x - x

    res[:, 0] = -x[:, 0]
    res[:, x.shape[1]-1] = x[:, x.shape[1]-2]

    return res

def LSP(param_E, param_d, param_lambda_L, param_lambda_S, param_nite, param_tol):
    '''

    :param param_E:
    :param param_d:
    :param param_lambda_L:
    :param param_lambda_S:
    :param param_nite:
    :param param_tol:
    :return:
    '''
    M = param_E(inv=True, data=param_d)
    nx, ny, nt = M.size()
    # M = torch.reshape(M, [nx * ny, nt])
    L = torch.zeros([nx * ny, nt], dtype=dtype)
    S = torch.zeros([nx * ny, nt], dtype=dtype)
    p_L = torch.zeros([nx * ny, nt], dtype=dtype)
    p_S = torch.zeros([nx * ny, nt], dtype=dtype)

    gamma = 0.5
    lambda_step = 1/10
    c = lambda_step/gamma

    loss = torch.zeros(param_nite, dtype=float)

    for itr in range(0, param_nite):

        # t1 = time()

        temp_data = torch.reshape(L+S, [nx, ny, nt])
        gradient = param_E(inv=True, data=param_E(inv=False, data=temp_data) - param_d)
        gradient = torch.reshape(gradient, [nx * ny, nt])
        y_L = L - gamma * gradient - gamma * p_L
        Par_L = c * y_L + p_L

        # t2 = time()
        # print('Run time (t2~t1) is %.4f s' % ((t2 - t1)))

        Ut, St, Vt = torch.svd(Par_L)
        temp_St = torch.diag(Project_inf(St, param_lambda_L))
        p_L = Ut.mm(temp_St).mm(Vt.T)
        L = L - gamma * gradient - gamma * p_L

        # t3 = time()
        # print('Run time (t3~t2) is %.4f s' % ((t3 - t2)))

        y_S = S - gamma * gradient - gamma * Wtxs(p_S)
        Par_S = c * Wxs(y_S) + p_S
        p_S = Project_inf(Par_S, param_lambda_S)

        # t4 = time()
        # print('Run time (t4~t3) is %.4f s' % ((t4 - t3)))

        S = S - gamma * gradient - gamma * Wtxs(p_S)

        # calculate loss
        # resk = param_E.mtimes(inv=False, data=torch.reshape(L+S, [nx, ny, nt])) - param_d
        # term1 = 1 / 2 * torch.norm(resk, p=2)**2
        # term2 = param_lambda_L * torch.norm(L, p='nuc')
        # term3 = param_lambda_S * torch.norm(torch.diff(S, dim=1), p=1)
        # loss[itr] = term1 + term2 + term3
        loss[itr] = 0

        # t5 = time()
        # print('Run time (t5~t4) is %.4f s' % ((t5 - t4)))

        print(' iteration: %d/%d, Loss: %f' % (itr+1, param_nite, loss[itr]))

    Ut, St, Vt = torch.svd(L)

    L = torch.matmul((Ut[:, 0]*St[0]).unsqueeze(1), Vt[:, 0].unsqueeze(0))
    L = torch.reshape(L, [nx, ny, nt])
    S = torch.reshape(S, [nx, ny, nt])

    return L, S, loss


