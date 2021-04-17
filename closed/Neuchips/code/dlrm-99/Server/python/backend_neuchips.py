#
# NEUCHIPS CONFIDENTIAL
#
# Copyright (C) 2018-2020 NEUCHIPS Corp.
# All Rights Reserved.
# Author: Tzu-Jen Julius Lo <tzujen_lo@neuchips.ai>
#         Brock Ko <brock_ko@neuchips.ai>
#
# The information and source code contained herein is the exclusive property
# of NEUCHIPS CORPORATION and may not be disclosed, examined or reproduced
# in whole or in part without explicit written authorization from the company.
#
import backend
from datetime import datetime
import math
import multiprocessing as mp
import numpy as np
import os
import torch

#
# NEUCHIPS proprietary packages
#
from utils_neuchips import RecAccel
from utils_neuchips import jprint
import utils_neuchips_lut
from utils_neuchips_ioctl import *
import struct
import fcntl

#
# RecAccel configurations
#
ACCELS_MAX_NUM = 2

#
# PCIE configurations
#
PCIE_VENDOR_ID = 0x1172
PCIE_PRODUCT_ID = 0xe003

#
# Global variables
#
accels = []
#
wpool = None
rfd = None


#
# BackendRecAccel - NEUCHIPS RecAccel, a DLRM inference accelerating solution
#
# version() - version
# name() - name
# load() - load LUT to RecAccel DRAM
# predict() - RecAccel backend prediction entry function
#
# do_predict() - major logic to interact with RecAccel worker/hardware
# collate_pre() - pre-process of collation
# collate_post() - collate dense and sparse torch to bytes
#
class BackendRecAccel(backend.Backend):

    def __init__(self):
        global accels
        global wpool
        global rfd

        jprint("recAccel - __init__")

        rfd = os.open("/dev/recaccel", os.O_RDWR)
        if rfd < 0:
            print("Failed to open RecAccel device")
            exit(-1)

        for i in range(ACCELS_MAX_NUM):
            accel = RecAccel(cid=i, batch_size=16, freq_mhz=200,
                             paral_dram=True)
            if accel.ready() is False:
                print("init accel", i, "failed")
                continue

            accels.append(accel)
            print("RecAccel-%d @%d MHz" % (accel.cid, accel.freq_mhz),
                  "ping-pong =", len(accel.buffs_pool) == 2)

        #
        # RecAccel input/output arguments
        #
        self.input_q = 4
        self.output_q = 7

        self.nbyte_per_inf = 308
        self.nbyte_per_batch = self.nbyte_per_inf * 16

        #
        # RecAccel capability
        #
        self.max_batchsize = 1024
        #
        wpool = mp.Pool(4)
        print("Backend max batch size = %d" % (self.max_batchsize))

    def version(self):
        return "1.0"

    def name(self):
        return "NEUCHIPS-RecAccel"

    # load - load LUT to RecAccel DRAM
    # @model_path: path of model
    #
    # Return: none
    def load(self, model_path, inputs=None, outputs=None):
        global accels

        def _write_embs(emb):
            conf = [[19, 10, 1, 23],
                    [0, 22, 4, 14, 13, 7, 17, 15, 24, 8, 25, 18, 12, 16, 5],
                    [21, 11, 2, 3],
                    [9, 20, 6]]

            for i in range(4):
                data = []

                jprint("    load DDR4%c" % (chr(ord('A')+i)))
                for idx in conf[i]:
                    data.extend(map(lambda f: f if f >= 0 else 256 + f,
                                    emb[idx].flip([1]).reshape(-1).tolist()))

                for accel in accels:
                    accel.dma_write(accel.mem_addr_ddr4[i], bytes(data),
                                    verify=False)
                del data

        jprint("recAccel - load")

        #
        # TODO: adaptively load LUT only when LUT is not loaded before
        #
        return self

        emb, bot, top = utils_neuchips_lut.collate_lut(model_path)

        _write_embs(emb)

        return self

    # predict - RecAccel backend prediction entry function
    # @ndata: NEUCHIPS collated data for RecAccel
    # @rsize: expected inference size
    #
    # Return: Predicted result in torch
    def predict(self, ndata, rsize):
        global wpool

        size = len(ndata) // self.nbyte_per_inf

        if size <= self.max_batchsize:
            return self.do_predict(ndata, size)

        #
        # Deploy multiprocessing pool
        #
        # @size = @quot * @self.max_batchsize + @resid
        #       = @quots + @resid
        quot = size // self.max_batchsize
        quots = quot * self.max_batchsize
        resid = size - quots
        # print("(%d) %d = %d + %d" %(rsize, size, quots, resid))
        nbyte_quots = quots * self.nbyte_per_inf

        _ndata = np.frombuffer(ndata[:nbyte_quots], dtype=np.byte)
        _ndata = list(map(bytes, np.array_split(_ndata, quot)))
        if resid != 0:
            _ndata.append(ndata[nbyte_quots:])

        jprint("pool mapping...")
        t = wpool.map(self.do_predict, _ndata)
        jprint("all done")
        res = torch.FloatTensor(rsize, 1)
        torch.cat(t, out=res)
        return res

    # do_predict - major logic to interact with RecAccel worker/hardware
    # @ndata: NEUCHIPS collated data for RecAccel
    # @size: inference size to predict; if None, calculate the size from @ndata
    #
    # Return: predicted result in torch
    def do_predict(self, ndata, size=None):
        global accels
        global rfd

        if size is None:
            size = len(ndata) // self.nbyte_per_inf

        #
        # Decrypt 16-infs aligned @ndata to build up splitting map
        #
        batch = len(ndata) // self.nbyte_per_batch
        npb = self.nbyte_per_batch
        trail = np.frombuffer(ndata[npb-1::npb], dtype=np.uint8)
        # FIXME: retrieve batch size from engine
        batch_size = 16
        valid = np.subtract(np.ones(batch, dtype=np.uint8) * batch_size, trail)
        # build up list for split
        smap = np.empty(batch * 2, dtype=np.uint8)
        smap[0::2] = valid
        smap[1::2] = trail

        # Acquire buffer
        uarg = bytearray(struct.pack(NCS_IOCTL_ARG, NFUNC_DLRM, 0, 0, 0))
        try:
            fcntl.ioctl(rfd, RECACCEL_IOCX_ACQUIRE_BUFFER, uarg)
            aid, handler, wr_addr, unuse = struct.unpack(NCS_IOCTL_ARG, uarg)
        except OSError:
            print("Failed IOCTL Acquire")
        accel = accels[aid]
        accel.dma_write(wr_addr, ndata)

        # Prediction
        uarg = bytearray(struct.pack(NCS_IOCTL_ARG, NFUNC_DLRM, handler,
                                     0, size))
        try:
            fcntl.ioctl(rfd, RECACCEL_IOCX_PREDICT, uarg)
            aid, handler, rd_addr, unuse = struct.unpack(NCS_IOCTL_ARG, uarg)
        except OSError:
            print("Failed IOCTL Predict")
        # Read out results and release buffer
        res = accel.dma_read_res_in_torch(rd_addr, size, self.output_q)

        # Release buffer
        uarg = struct.pack(NCS_IOCTL_ARG, NFUNC_DLRM, handler, 0, 0)
        try:
            fcntl.ioctl(rfd, RECACCEL_IOCX_RELEASE_BUFFER, uarg)
        except OSError:
            print("Failed IOCTL Release")

        return torch.cat(res.split(smap.tolist())[0::2])

    # collate_pre - pre-process of collation
    # @dense: torch of dense
    # @sparse: torch of sparse
    #
    # Return: pair of pre-processed torch
    def collate_pre(self, dense, sparse):

        def __collate_sps(s):
            zerox16 = np.zeros(16, dtype=np.uint8)
            res = []

            s_t = s.T
            for inf in s_t:
                for i in range(26):
                    res.extend(list(int(inf[i].item()).to_bytes(8, 'little',
                                                                signed=False)))
                res.extend(np.repeat(zerox16, 3))

            new_s_t = np.asarray(res, dtype=np.uint8)
            new_s_t = np.reshape(new_s_t, (s.shape[1], 256))

            return torch.from_numpy(new_s_t.T)

        dns = torch.clamp(torch.floor(dense.to(torch.float64) *
                                      pow(2, self.input_q) + 0.5),
                          0, 255).to(torch.uint8)
        sps = __collate_sps(sparse)
        return dns, sps

    # collate_post - collate dense and sparse torch to bytes
    # @d: pre-processed dense in torch
    # @s: pre-processed sparse in torch
    #
    # Return: collated metadata in bytes
    def collate_post(self, d, s):

        infs = d.shape[0]
        s = s.T
        assert(infs == s.shape[0])

        base = 0
        _infs = infs

        d_np = np.asarray(d, dtype=np.uint8)
        s_np = np.asarray(s, dtype=np.uint8)

        r = _infs % 16
        if r != 0:
            r = 16 - r
            d_np = np.concatenate((d_np, np.zeros((r, d_np.shape[1]),
                                                  dtype=np.uint8)))
            s_np = np.concatenate((s_np, np.zeros((r, s_np.shape[1]),
                                                  dtype=np.uint8)))

        d_loop = int(d_np.shape[0]/16)
        res = np.zeros((4928*d_loop,), np.uint8)

        d_t = d_np.T
        ptr_d = 0

        # dense
        iis = 0
        for i in range(d_loop):
            base = ptr_d
            iit = iis + 16
            # for j in range(13):
            #    res[base:base+16] = d_t[j][iis:iit]
            #    base += 64
            # by chiung begin
            res[base:base+16] = d_t[0][iis:iit]
            base += 64
            res[base:base+16] = d_t[1][iis:iit]
            base += 64
            res[base:base+16] = d_t[2][iis:iit]
            base += 64
            res[base:base+16] = d_t[3][iis:iit]
            base += 64
            res[base:base+16] = d_t[4][iis:iit]
            base += 64
            res[base:base+16] = d_t[5][iis:iit]
            base += 64
            res[base:base+16] = d_t[6][iis:iit]
            base += 64
            res[base:base+16] = d_t[7][iis:iit]
            base += 64
            res[base:base+16] = d_t[8][iis:iit]
            base += 64
            res[base:base+16] = d_t[9][iis:iit]
            base += 64
            res[base:base+16] = d_t[10][iis:iit]
            base += 64
            res[base:base+16] = d_t[11][iis:iit]
            base += 64
            res[base:base+16] = d_t[12][iis:iit]
            # by chiung end

            ptr_d += 4928
            iis += 16

        # sparse
        ptr_d = 0
        base = 0
        while _infs > 0:
            ptr_s = ptr_d + 832
            ptr_d = ptr_s + 4096
            res[ptr_s:ptr_d] = s_np[base:base+16].flatten()

            _infs -= 16
            base += 16

        # for i in range(0, 52, 4):
        #     print("0x%02x" %(i), res[i*16:i*16+16])

        res[-1] = r
        return bytes(res)
