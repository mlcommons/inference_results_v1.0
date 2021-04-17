#
# NEUCHIPS CONFIDENTIAL
#
# Copyright (C) 2018-2020 NEUCHIPS Corp.
# All Rights Reserved.
# Author: Chiung-Liang Lin <chiungliang_lin@neuchips.ai>
#
# The information and source code contained herein is the exclusive property
# of NEUCHIPS CORPORATION and may not be disclosed, examined or reproduced
# in whole or in part without explicit written authorization from the company.
#
import numpy as np
import torch


def check_sign_q(w):

    # define 8-bit range
    q_max = 127
    q_min = -128

    # check input data
    max_v = torch.max(w)
    min_v = torch.min(w)
    # print('(max, min) = ' + str(max_v) + ', ' + str(min_v))

    if max_v <= 0:
        max_v = 1e-7
    if min_v >= 0:
        min_v = 1e-7

    # scale
    max_scale = q_max / max_v
    min_scale = q_min / min_v

    #
    if max_scale < min_scale:
        sel_scale = max_scale
    else:
        sel_scale = min_scale

    #
    log2_sel_scale = torch.log2(sel_scale)

    #
    return (int)(torch.floor(log2_sel_scale))


def quant_pow2(x, q_frac):

    x = torch.floor(x * pow(2, q_frac) + 0.5)
    q_x = x.to(torch.int8)

    return q_x


def idx_remap(dim):

    '''
    -> low triangle

    ex: 5x5 array
    -  -  -  -  -
    0  -  -  -  -
    1  2  -  -  -
    3  4  5  -  -
    6  7  8  9  -
    '''
    idx = 0
    p = []
    for y in range(dim):
        for x in range(y):
            p.append([y, x, idx])
            idx += 1

    '''
    -> sorted by x, then by y

    -  -  -  -  -
    0  -  -  -  -
    1  4  -  -  -
    2  5  7  -  -
    3  6  8  9  -
    '''
    p = sorted(p, key=lambda s: (s[1] << 16) + s[0])

    '''
    -> merge {n, dim-n}

    -  -  -  -  -
    0  -  -  -  -
    1  5  -  -  -
    2  6  8  -  -
    3  7  9  4  -

    0  2  3  1  <-- order
    '''

    idx = []
    dim = dim - 1
    dim_div2 = (dim >> 1)
    for i in range(dim_div2):
        for j in range(len(p)):
            if p[j][1] == i:  # x = i
                idx.append(p[j][2])
            elif p[j][1] == dim-i-1:  # x = dim-i
                idx.append(p[j][2])

    return idx


def merge_wb(w, b, q_in, q_w, q_b):

    #
    duplicate_pow = max(0, q_w + q_in - q_b - 7)

    #
    duplicate_b = b.reshape(b.shape[0], 1)

    for i in range(duplicate_pow):
        duplicate_b = torch.cat((duplicate_b, duplicate_b), 1)

    #
    wb = torch.cat((w, duplicate_b), 1)

    '''
    print("------------>")
    print(w.shape)
    print(b.shape)
    print(wb.shape)
    '''

    return wb


def hw_wb(w):

    # 2d -> 1d
    dim_y = w.shape[0]
    dim_x = w.shape[1]
    w = w.reshape(dim_x*dim_y)

    results = []
    for j in range(0, dim_y, 256):
        for i in range(dim_x):
            for sub_j in range(255, -1, -1):
                temp_j = j + sub_j
                if temp_j < dim_y:
                    results.append(w[temp_j*dim_x+i])
                else:
                    results.append(torch.tensor(0, dtype=torch.int8))

    # print("result", type(results), len(results), type(results[0]))

    '''
    for i in range(len(results)):
        print(results[i], end='\t')
        if i%256==255:
            print('\n')
    '''

    return results


def collate_lut(path='./model/dlrm_terabyte.pytorch'):

    # read params from pt
    params_torch = torch.load(path)
    params = params_torch['state_dict']

    if len(params['emb_l.0.weight']) == 9980333:
        # q format
        # terabyte 0.875
        q_in_bot_mlp = [4, 6, 6]
        q_in_top_mlp = [4, 7, 7, 7]
        print("fake_terabyte")
    else:
        # terabyte
        q_in_bot_mlp = [4, 3, 4]
        q_in_top_mlp = [4, 5, 5, 5, 5]
        print("terabyte")

    print("Read params")

    # quantize & classify
    emb = []
    bot_mlp = []
    top_mlp = []
    # q_emb = []
    q_bot_mlp = []
    q_top_mlp = []

    idx = 0
    num_emb = 0
    dim_bot_mlp_last = 0
    for key in params:

        q = check_sign_q(params[key])
        # print(q)

        if key.find("emb") != -1:
            num_emb += 1
            emb.append(quant_pow2(params[key], q))
            # q_emb.append(q)
        elif key.find("bot") != -1:
            bot_mlp.append(quant_pow2(params[key], q))
            q_bot_mlp.append(q)
        else:
            top_mlp.append(quant_pow2(params[key], q))
            q_top_mlp.append(q)

        idx = idx+1

    print("quantize & classify parameters")

    # release mem
    del params
    del params_torch

    '''
    print("emb")
    for key in emb:
        print(key.shape)
    print("bot_mlp")
    for key in bot_mlp:
        print(key.shape)

    print("top_mlp")
    for key in top_mlp:
        print(key.shape)
    '''

    '''
    # sort emb
    hw_emb = sorted(emb, key = lambda s: s.shape[0], reverse=True)

    print("sorted emb")
    '''
    hw_emb = emb

    for key in hw_emb:
        print(key.shape)

    # release mem
    del emb

    #
    hw_bot_mlp = []
    hw_top_mlp = []

    idx = 0
    for i in range(0, len(bot_mlp), 2):
        hw_bot_mlp.extend(hw_wb(merge_wb(bot_mlp[i], bot_mlp[i+1],
                                         q_in_bot_mlp[idx], q_bot_mlp[i],
                                         q_bot_mlp[i+1])))
        idx += 1
        # input("Press Enter to continue...")

    idx = 0
    for i in range(0, len(top_mlp), 2):

        if i == 0:  # reorder to fix output of interaction

            dim_bot_mlp_last = bot_mlp[-1].shape[0]
            idx_hw_top_mlp = np.array(idx_remap(num_emb+1))
            idx_hw_top_mlp_offset = idx_hw_top_mlp + dim_bot_mlp_last
            idx_reorder = np.concatenate([np.array(range(dim_bot_mlp_last)),
                                          idx_hw_top_mlp_offset])

            new_w = top_mlp[i][:, idx_reorder]
            wb = merge_wb(new_w, top_mlp[i+1], q_in_top_mlp[idx],
                          q_top_mlp[i], q_top_mlp[i+1])

        else:
            wb = (merge_wb(top_mlp[i], top_mlp[i+1], q_in_top_mlp[idx],
                           q_top_mlp[i], q_top_mlp[i+1]))

        hw_top_mlp.extend(hw_wb(wb))
        idx += 1
        # input("Press Enter to continue...")

    """
    hw_emb
    hw_bot_mlp
    hw_top_mlp
    """

    return (hw_emb, hw_bot_mlp, hw_top_mlp)
