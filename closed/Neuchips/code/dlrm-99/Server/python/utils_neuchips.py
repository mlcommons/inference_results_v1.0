#
# NEUCHIPS CONFIDENTIAL
#
# Copyright (C) 2018-2020 NEUCHIPS Corp.
# All Rights Reserved.
# Author: Tzu-Jen Julius Lo <tzujen_lo@neuchips.ai>
#
# The information and source code contained herein is the exclusive property
# of NEUCHIPS CORPORATION and may not be disclosed, examined or reproduced
# in whole or in part without explicit written authorization from the company.
#
import ctypes
from datetime import datetime
import math
import multiprocessing as mp
import numpy as np
import os
import queue
import threading
import time
import torch
import psutil

#
# FPGA control registers layout
#
REG_ADDR_BASE = 0x4001000
REG_ADDR_CTRL = REG_ADDR_BASE
REG_ADDR_BUFFER_SET = REG_ADDR_BASE + 0x8
REG_ADDR_BATCH_CTRL = REG_ADDR_BASE + 0x14

REG_ADDR_BATCH_BITMAP = REG_ADDR_BASE + 0x18
REG_ADDR_BATCH_BITMAP_HI = REG_ADDR_BASE + 0x1c
REG_ADDR_DMA_ITER = REG_ADDR_BASE + 0x20
REG_ADDR_DMA_ITER_RES = REG_ADDR_BASE + 0x24

REG_ADDR_BATCH_START = REG_ADDR_BASE + 0xc
REG_ADDR_INIT_DL_CMPLT = REG_ADDR_BASE + 0x130
REG_ADDR_BATCH_FINISH = REG_ADDR_BASE + 0x134

LSHFT_BCHCTRL_BURST_LENGTH = 16

MASK_CTRL_RESET = 0x10000
MASK_BCHCTRL_BATCH_SIZE = 0x7f

#
# PCIE configurations
#
PCIE_VENDOR_ID = 0x1172
PCIE_PRODUCT_ID = 0xe003

USER_BAR = 4

DMA_CHUNK_SIZE = 0x80000000

bytes_per_inf = 308  # for batch 16
bytes_per_dma_iter = 16384
bytes_per_dma_cycle = 64

settings = [
    0x1a801241, 0x0d036666, 0x35028700, 0x350c8700,
    0x008015e4, 0x00d04c5e, 0x00000000, 0x0fa4a133,
    0x03012201, 0x01002030, 0x01230000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
    0x0307b1a0, 0x00000000, 0x03ffffff, 0x00000000,
    0x03ffffff, 0x00000000, 0x03ffffff, 0x00000000,
    0x03ffffff, 0x00000000, 0x03ffffff, 0x00000000,
    0x0000000f, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x33400333, 0x32342333, 0x33232414, 0x00000044,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x33400333, 0x32342333, 0x33232414, 0x00000044,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x33400333, 0x32342333, 0x33232414, 0x00000044,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x33400333, 0x32342333, 0x33232414, 0x00000044,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x478e2280, 0x31b27500, 0x31d43980,
    0x34c3a280, 0x000013fd, 0xe99c5300, 0x000008a0,
    0x0000137e, 0x00000000, 0x31057d80, 0x2e9eac00,
    0x000013ef, 0x00000000, 0x34eb3600, 0x00001277,
    0x000013f9, 0x00000ea7, 0x000013e1, 0x00000000,
    0x25fbab80, 0x00000000, 0x304b3b00, 0x47da6400,
    0x00001312, 0x000013bd, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000007, 0x47f3ec00, 0x47f8ee00, 0x31e2e980,
    0x00000000, 0x00000000, 0x00000000, 0x00000190,
    0x50000020, 0x10c00000, 0x28000000, 0xf0400040,
    0x00000000, 0x7ff00030, 0x7800000a, 0x00000000,
    0x78000006, 0x00000000, 0x50100020, 0x10c40000,
    0x28000000, 0xf0400040, 0x00003800, 0x7ff00030,
    0x78000007, 0x00000000, 0x40000020, 0x10a40000,
    0x30000020, 0x10400040, 0x04007010, 0x1ff00030,
    0x7a000012, 0x00000000, 0x50000010, 0x10c40000,
    0x20000010, 0x10500040, 0x04087808, 0x1ff00030,
    0x0014000f, 0x0ff00030, 0x40000020, 0x10a00000,
    0x2800001e, 0x10200040, 0x000c800f, 0x0ff00030,
    0x2800001e, 0x10200040, 0x001b800f, 0x0ff00030,
    0x50100020, 0x10a00000, 0x2800001e, 0x10200040,
    0x50300020, 0x10a40000, 0x2800001e, 0x10200040,
    0x0023000f, 0x0ff00030, 0x50200020, 0x10a00000,
    0x40000020, 0x11000000, 0x30000040, 0x10600040,
    0x042a8020, 0x1ff00030, 0x78000012, 0x00000000,
    0x044a9020, 0x1ff00030, 0x40100020, 0x11000000,
    0x30000040, 0x10600040, 0x043a8820, 0x1ff00030,
    0x30000040, 0x10600040, 0x045a9820, 0x1ff00030,
    0x40200020, 0x11000000, 0x30000040, 0x10600040,
    0x20000040, 0x10600040, 0x046aa020, 0x1ff00030,
    0x7800000a, 0x00000000, 0x40300020, 0x11040000,
    0x50100020, 0x11240000, 0x20000040, 0x10600040,
    0x047aa820, 0x1ff00030, 0x50000020, 0x11200000,
    0x40000020, 0x11040000, 0x30000020, 0x10600040,
    0x048ab010, 0x1ff00030, 0x78000006, 0x00000000,
    0x50000000, 0x21340000, 0x20000010, 0x10700040,
    0x0092b808, 0x1ff00030, 0x78000007, 0x00000000
]

CARD_DEV_PATH = "/dev/altera_pcie"

#
# FPGA DDR4 layout
#
MEM_ADDR_ONCHIP = 0x00000000
MEM_ADDR_DDR4A = 0x800000000
MEM_ADDR_DDR4B = 0xa00000000
MEM_ADDR_DDR4C = 0xc00000000
MEM_ADDR_DDR4D = 0xe00000000

MEM_ADDR_SETTINGS = MEM_ADDR_ONCHIP
MEM_ADDR_BOT = MEM_ADDR_ONCHIP + 0x240
MEM_ADDR_TOP = MEM_ADDR_ONCHIP + 0x0

MEM_ADDR_RES = MEM_ADDR_ONCHIP
MEM_ADDR_INFS = MEM_ADDR_DDR4A + 0x147f3ec00
MEM_ADDR_RES_2 = MEM_ADDR_ONCHIP + 0x800
MEM_ADDR_INFS_2 = MEM_ADDR_DDR4A + 0x147f8ee00

#
# Enumerations
#
CARD_STAT_INVALID = -1
CARD_STAT_READY = 0
CARD_STAT_OCCUPIED = 1


def jprint(*args):
    return
    pid = os.getpid()
    iid = threading.get_ident()
    print("[%s] (%d:%03d)" % (datetime.now().strftime("%H:%M:%S.%f"), pid,
                              iid % 1000), *args)


def trace_begin(*args):
    ps = psutil.Process()
    print("%16s-%-5d [%03d] .... %.6f: tracing_mark_write: B|%d|%s" %
          (ps.name()[:16], ps.pid, ps.cpu_num(),
           time.clock_gettime(time.CLOCK_MONOTONIC), ps.ppid(), *args))
    print(time.clock_gettime(time.CLOCK_MONOTONIC))


def trace_end():
    ts = time.clock_gettime(time.CLOCK_MONOTONIC)
    ps = psutil.Process()
    print("%16s-%-5d [%03d] .... %.6f: tracing_mark_write: E" %
          (ps.name()[:16], ps.pid, ps.cpu_num(), ts))


#
# RecAccelBuff - RecAccel in/output buffer
#
class RecAccelBuff():

    def __init__(self, bid, addr_in, addr_out):
        self.bid = bid
        self.addr_in = addr_in
        self.addr_out = addr_out
        self.sync = mp.Event()
        # no need to wait for reading at the first time
        self.sync.set()


#
# RecAccel - NEUCHIPS RecAccel
#
# reg_read() - read a word from a register
# reg_read_locked() - read a word with @self.hwlock locked
# reg_write() - write a word to a register
# reg_write_locked() - write a word with @self.hwlock locked
#
# reg_write_read() - write a word and then verify it
# reg_write_bitwise_pulse() - raise a pulse
# reg_write_bitwise() - bitwise write to a register
#
# dma_read() - read @size bytes from FPGA DDR4
# dma_write() - write @data to FPGA DDR4
# dma_read_res_in_torch() - read results in troch
# mem_dump() - memory dump for debugging
#
# ready() - readiness of RecAccel
# hardware_init() - set up hardware initial configuration
# rundtrip_time() - estimated RecAccel inference time
# predict() - major function to interact with RecAccel
#
class RecAccel():

    def __init__(self, cid, init=True, batch_size=16, freq_mhz=200,
                 paral_dram=True):
        super().__init__()

        self.cid = cid
        self.status = CARD_STAT_INVALID

        if os.path.exists(CARD_DEV_PATH + str(cid)) is False:
            return

        self.clib = ctypes.cdll.LoadLibrary(
            "./python/cpython/lib/terasic_pcie_qsys.so")
        self.clib.PCIE_Read32.argtypes = (ctypes.c_uint, ctypes.c_int,
                                          ctypes.c_uint64,
                                          ctypes.POINTER(ctypes.c_uint))
        self.clib.PCIE_DmaRead.argtypes = (ctypes.c_uint, ctypes.c_uint64,
                                           ctypes.POINTER(ctypes.c_uint8),
                                           ctypes.c_uint)
        self.clib.PCIE_DmaWrite.argtypes = (ctypes.c_uint, ctypes.c_uint64,
                                            ctypes.POINTER(ctypes.c_uint8),
                                            ctypes.c_uint)

        # get a pcie file descriptor
        self.fd = self.clib.PCIE_Open(PCIE_VENDOR_ID, PCIE_PRODUCT_ID, cid)
        if self.fd == 0:
            return

        # for RecAccel HW synchronization
        self.hwlock = mp.Lock()

        self.batch_size = batch_size
        self.freq_mhz = freq_mhz
        self.mem_addr_ddr4 = [MEM_ADDR_DDR4A, MEM_ADDR_DDR4B,
                              MEM_ADDR_DDR4C, MEM_ADDR_DDR4D]
        self.init_dl_tol_cnt = 1000
        self.batch_extra_tol_cnt = 100

        #
        self.status = CARD_STAT_READY
        if init is True:
            self.status = self.hardware_init(MEM_ADDR_SETTINGS)

        # in/out buffers
        self.buffs_pool = []
        self.buffs_pool.append(RecAccelBuff(0, MEM_ADDR_INFS,
                                            MEM_ADDR_RES))
        self.buffs_pool.append(RecAccelBuff(1, MEM_ADDR_INFS_2,
                                            MEM_ADDR_RES_2))
        self.buffs = mp.Queue()
        for b in range(len(self.buffs_pool)):
            self.buffs.put(b)

        self.paral_dram = paral_dram

    # reg_read - read a word from a register
    #
    # Return: a word
    def reg_read(self, addr, _fd=None):
        res = ctypes.c_uint(0)
        with self.hwlock:
            self.clib.PCIE_Read32(_fd or self.fd, USER_BAR, addr,
                                  ctypes.byref(res))
        # print(addr, res.value)
        return res.value

    # reg_read_locked - read a word with @self.hwlock locked
    #
    # Return: a word
    def reg_read_locked(self, addr, _fd=None):
        res = ctypes.c_uint(0)
        self.clib.PCIE_Read32(_fd or self.fd, USER_BAR, addr,
                              ctypes.byref(res))
        return res.value

    # reg_write - write a word to a register
    # @addr: address to write
    # @data: a word
    # @_fd: associated file descriptor
    #
    # Return: status of writing
    def reg_write(self, addr, data, _fd=None):
        with self.hwlock:
            self.clib.PCIE_Write32(_fd or self.fd, USER_BAR, addr, data)

    # reg_write_locked - write a word with @self.hwlock locked
    # @addr: address to write
    # @data: a word
    # @_fd: associated file descriptor
    #
    # Return: status of writing
    def reg_write_locked(self, addr, data, _fd=None):
        return self.clib.PCIE_Write32(_fd or self.fd, USER_BAR, addr, data)

    # reg_write_read - write a word and then verify it
    # @addr: address to write
    # @data: a word
    # @_fd: associated file descriptor
    #
    # Return: none
    def reg_write_read(self, addr, data, _fd=None):
        fd = _fd or self.fd
        with self.hwlock:
            self.reg_write_locked(addr, data, fd)
            readback = self.reg_read_locked(addr, fd)
        if readback != data:
            print("write %08x: 0x%08x (expect 0x%08x)" %
                  (addr, readback, data))

    # reg_write_bitwise_pulse - raise a pulse
    # @addr: address to write
    # @data: a word; each outstanding bit has a pulse simultaneously
    # @_fd: associated file descriptor
    #
    # Return: none
    def reg_write_bitwise_pulse(self, addr, data, _fd=None):
        fd = _fd or self.fd
        with self.hwlock:
            orig = self.reg_read_locked(addr, fd)
            self.reg_write_locked(addr, (orig | data), fd)
            self.reg_write_locked(addr, (orig & (~data)), fd)

    # reg_write_bitwise - bitwise write to a register
    # @addr: address to write
    # @data: a word, each outstanding bit is written
    # @_fd: associated file descriptor
    #
    # Return: none
    def reg_write_bitwise(self, addr, data, _fd=None):
        fd = _fd or self.fd
        with self.hwlock:
            orig = self.reg_read_locked(addr, fd)
            self.reg_write_locked(addr, (orig | data), fd)

    # dma_read - read @size bytes from FPGA DDR4
    #
    # Return: a byte list
    def dma_read(self, addr, size, log=True):
        jprint("dma_read  0x%09x %u" % (addr, size))

        res = []
        _addr = addr
        _size = size

        # Loop by @DMA_CHUNK_SIZE
        while _size > 0:
            chunk_size = min(_size, DMA_CHUNK_SIZE)
            data = (ctypes.c_uint8 * chunk_size)()
            ctypes.cast(data, ctypes.POINTER(ctypes.c_uint8))

            with self.hwlock:
                self.clib.PCIE_Write32(self.fd, USER_BAR, REG_ADDR_BASE + 0x10,
                                       0x8)
                ret = self.clib.PCIE_DmaRead(self.fd, _addr, data, chunk_size)
            if ret != 1:
                jprint("DMA read  0x%09x %12lu bytes failed" %
                       (_addr, chunk_size))
                exit(1)

            _addr += chunk_size
            _size -= chunk_size
            res.extend(data)
            del data

        if log is True:
            jprint("DMA read  0x%09x %12lu bytes" % (addr, size))
        return res

    # dma_write - write @data to FPGA DDR4
    # @addr: address to write
    # @data: data in bytes
    # @verify: true to perform read back verification
    #
    # Return: None
    def dma_write(self, addr, data, verify=False):
        jprint("dma_write 0x%09x %u" % (addr, len(data)))

        if len(data) % 4 != 0:
            print("DMA write 0x%09x %12lu bytes is not word-sized" %
                  (addr, len(data)))
            return

        assert(isinstance(data, bytes))

        _addr = addr
        _data = data
        _size = len(data)

        # Loop by @DMA_CHUNK_SIZE
        while _size > 0:
            chunk_size = min(_size, DMA_CHUNK_SIZE)
            _d = _data[:chunk_size]
            chars = (ctypes.c_uint8 * chunk_size)()
            ctypes.memmove(ctypes.addressof(chars), _d, chunk_size)
            del _d
            chars_len = ctypes.sizeof(chars)

            # Issue DMA operation
            with self.hwlock:
                self.clib.PCIE_Write32(self.fd, USER_BAR, REG_ADDR_BASE + 0x10,
                                       0x8)
                ret = self.clib.PCIE_DmaWrite(self.fd, _addr, chars, chars_len)
            if ret != 1:
                jprint("DMA write 0x%09x %12lu bytes failed" %
                       (_addr, chars_len))
                exit(1)

            if verify is True:
                res = self.dma_read(_addr, chars_len, log=False)
                for i in range(chars_len):
                    if res[i] == chars[i]:
                        continue

                    print("\nDMA verify error 0x%09x: %02x != %02x\n" %
                          (_addr+i, res[i], chars[i]))
                    print("DMA write 0x%09x %12lu bytes failed" %
                          (_addr, chars_len))
                    exit(1)

            del chars

            _addr += chunk_size
            _data = _data[chunk_size:]
            _size -= chunk_size

        jprint("DMA write 0x%09x %12lu bytes" % (addr, len(data)))

    # dma_read_res_in_torch - read results in troch
    # @addr: address of results to read
    # @size: expected size in byte
    # @q: de-quantization order
    #
    # Return: inference results in torch
    def dma_read_res_in_torch(self, addr, size, q):
        # 4-bytes alignment DRAM access
        words = 4
        rdata = self.dma_read(addr, words * math.ceil(size/words))

        res = np.asarray(rdata[:size]).astype(np.float32)
        res /= (1 << q)
        res = torch.from_numpy(res.reshape(size, 1))
        return res

    # mem_dump - memory dump for debugging
    #
    # Return: none
    def mem_dump(self, data, size):
        print(size)
        for i in range(0, size, 16):
            print("%02x %02x %02x %02x %02x %02x %02x %02x  \
                   %02x %02x %02x %02x %02x %02x %02x %02x" %
                  (data[i], data[i+1], data[i+2], data[i+3],
                   data[i+4], data[i+5], data[i+6], data[i+7],
                   data[i+8], data[i+9], data[i+10], data[i+11],
                   data[i+12], data[i+13], data[i+14], data[i+15]))

    # ready - readiness of RecAccel
    #
    # Return: true if RecAccel is available, otherwise false
    def ready(self):
        return (self.status == CARD_STAT_READY)

    # hardware_init - set up hardware initial configuration
    # @addr: address for config.
    #
    # Return: none
    def hardware_init(self, addr):

        def _word_to_chars(word):
            res = []
            wlen = len(word)
            for i in range(0, wlen, 8):
                for j in range(7, -1, -1):
                    t = word[i+j]
                    for _ in range(4):
                        res.append(t & 0xff)
                        t >>= 8
            return bytes(res)

        # write setting
        self.dma_write(addr, _word_to_chars(settings))

        # reset
        self.reg_write_bitwise_pulse(REG_ADDR_CTRL, MASK_CTRL_RESET)

        # kick-off FPGA to download settings
        self.reg_write_bitwise(REG_ADDR_CTRL, 0x1)
        # wait for completion
        cmplt = 0
        cnt = 0
        while cmplt == 0:
            cnt += 1
            if cnt > self.init_dl_tol_cnt:
                return CARD_STAT_INVALID
            cmplt = self.reg_read(REG_ADDR_INIT_DL_CMPLT)
        # clear by software
        self.reg_write(REG_ADDR_INIT_DL_CMPLT, 0)
        return CARD_STAT_READY

    # rundtrip_time - estimated RecAccel process time
    # @infs: inference count
    #
    # Return: time in second
    def rundtrip_time(self, infs):
        bursts = math.ceil(infs/self.batch_size)
        kilo_cycles = (bursts // 2) * 21
        if bursts % 2 == 1:
            kilo_cycles += 14
        # second per kilo cycles = 1000 * (1 / (freq_mhz() * 1000000))
        return (kilo_cycles * (1 / (self.freq_mhz * 1000)))

    # predict - major function to interact with RecAccel
    # @bid: buffer id for RecAccel usage
    # @batch_size: batch size
    # @burst_len: burst length
    # @fd: default file descriptor is acquired in RecAccel instance process,
    #      allowing other process to deploy its own @fd for parallelism
    #
    # Return: none
    def predict(self, bid, batch_size=16, burst_len=1, _fd=None):

        def _init(bid, batch_size, burst_len, fd):
            # select input buffer
            self.reg_write(REG_ADDR_BUFFER_SET, bid, fd)

            data = burst_len << LSHFT_BCHCTRL_BURST_LENGTH
            data |= (batch_size & MASK_BCHCTRL_BATCH_SIZE)
            self.reg_write_read(REG_ADDR_BATCH_CTRL, data, fd)

            # set batch bitmap
            data = (1 << batch_size) - 1
            self.reg_write_read(REG_ADDR_BATCH_BITMAP, data, fd)
            # TODO: batch_size > 32
            self.reg_write_read(REG_ADDR_BATCH_BITMAP_HI, 0, fd)

            # set DMA cycles
            data = bytes_per_inf * self.batch_size
            self.reg_write_read(REG_ADDR_DMA_ITER,
                                math.ceil(data / bytes_per_dma_iter), fd)
            data %= bytes_per_dma_iter
            self.reg_write_read(REG_ADDR_DMA_ITER_RES,
                                math.ceil(data / bytes_per_dma_cycle), fd)

        def _start(fd):
            self.reg_write(REG_ADDR_BATCH_START, 1, fd)

        # _wait - wait RecAccel for completion
        #
        # Return: none
        def _wait(self, n_infs, fd):
            done = 0
            cnt = 0
            time.sleep(self.rundtrip_time(n_infs))
            while done != 1:
                cnt += 1
                if cnt > self.batch_extra_tol_cnt:
                    print("RecAccel-%d" % (self.cid),
                          "failed to predict of time-out")
                    r = self.reg_read(REG_ADDR_BASE + 0x4, fd)
                    print("cycles", r)

                    self.status = CARD_STAT_INVALID
                    res = np.ones(n_infs).astype(float)
                    res /= (1 << 7)  # an arbitrary quantization factor
                    res = torch.from_numpy(res.reshape(n_infs, 1))
                    return res

                time.sleep(0.000010)
                done = self.reg_read(REG_ADDR_BATCH_FINISH, fd)
            # clear by software
            self.reg_write(REG_ADDR_BATCH_FINISH, 0, fd)

            # Debug logging
            # r = self.reg_read(REG_ADDR_BASE + 0x4, fd)
            # print("cycles", r)

        # Entry of prediction
        jprint("RecAccel-%d" % (self.cid), batch_size, burst_len)

        fd = _fd or self.fd
        _init(bid, batch_size, burst_len, fd)
        _start(fd)
        _wait(self, batch_size * burst_len, fd)


if __name__ == "__main__":
    acard = RecAccel(0, init=False)
    _data = acard.dma_read(MEM_ADDR_RES, 128)
    acard.mem_dump(_data, 128)
