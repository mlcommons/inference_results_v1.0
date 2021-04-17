# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import torch
import math

constant = 0.00001


class DALIInferencePipeline(Pipeline):
    def __init__(self,
                 device_id,
                 num_threads,
                 resample_range: list,
                 sample_rate=16000,
                 window_size=0.02,
                 window_stride=0.01,
                 window="hann",
                 normalize="per_feature",
                 n_fft=None,
                 preemph=0.97,
                 nfilt=64,
                 lowfreq=0,
                 highfreq=0,
                 log=True,
                 dither=constant,
                 pad_to=8,
                 max_duration=15.0,
                 frame_splicing=3,
                 batch_size=1,
                 total_samples=16,
                 audio_fp16_input=True,
                 device='gpu'):
        super().__init__(batch_size, num_threads, device_id,
                         exec_async=True, exec_pipelined=True, seed=12, prefetch_queue_depth=1)

        self._dali_init_log(locals())
        if torch.distributed.is_initialized():
            shard_id = torch.distributed.get_rank()
            n_shards = torch.distributed.get_world_size()
        else:
            shard_id = 0
            n_shards = 1

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.audio_fp16_input = audio_fp16_input
        self.total_samples = total_samples
        self.win_length = int(sample_rate * window_size)  # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        self.highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None

        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window_tensor

        self.lowfreq = lowfreq
        self.log = log
        self.device = device

        win_unpadded = self.window.tolist()
        win_padded = win_unpadded + [0] * (self.n_fft - len(win_unpadded))

        print("self.n_fft = {}".format(self.n_fft))
        print("self.hop_length = {}".format(self.hop_length))
        print("self.win_length = {}".format(self.win_length))
        print("self.window_tensor = {}".format(self.window))
        print("self.sample_rate = {}".format(self.sample_rate))
        print("self.window_size = {}".format(self.window_size))
        print("self.window_stride = {}".format(self.window_stride))
        print("self.lowfreq = {}".format(self.lowfreq))
        print("self.device = {}".format(self.device))

        self.extsrc = ops.ExternalSource(name="INPUT_0", device=self.device, no_copy=True)

        self.preemph = ops.PreemphasisFilter(preemph_coeff=preemph, device=self.device)

        self.spectrogram = ops.Spectrogram(device=self.device,
                                           nfft=self.n_fft,
                                           center_windows=True,
                                           window_fn=win_padded,
                                           window_length=len(win_padded),
                                           window_step=self.hop_length
                                           )
        self.mel_fbank = ops.MelFilterBank(device=self.device,
                                           sample_rate=self.sample_rate,
                                           nfilter=self.nfilt,
                                           freq_high=self.highfreq,
                                           freq_low=self.lowfreq,
                                           normalize=normalize
                                           )

        self.log_features = ops.ToDecibels(device=self.device, multiplier=np.log(10), reference=1.0,
                                           cutoff_db=math.log(1e-20))

        self.get_shape = ops.Shapes(device=self.device)

        self.normalize = ops.Normalize(axes=[0], device=self.device, ddof=1)

        self.pad = ops.Pad(axes=[0, 1], fill_value=0, shape=[502, 240], device=self.device)

        # Frame splicing
        self.splicing_transpose = ops.Transpose(device=self.device, perm=[1, 0])
        self.splicing_reshape = ops.Reshape(device=self.device, rel_shape=[-1, self.frame_splicing])
        self.splicing_pad = ops.Pad(axes=[0], fill_value=0, align=self.frame_splicing, shape=[1], device=self.device)

        self.to_float16 = ops.Cast(dtype=types.FLOAT16, device=self.device)
        self.to_float32 = ops.Cast(dtype=types.FLOAT, device=self.device)

        self.samples_done = 0

    @classmethod
    def from_config(cls, device_id, batch_size, total_samples, num_threads, device: str, audio_fp16_input: bool, config: dict):
        return cls(device=device,
                   device_id=device_id,
                   batch_size=batch_size,
                   total_samples=total_samples,
                   num_threads=num_threads,
                   sample_rate=config.get('sample_rate', 16000),
                   resample_range=[.9, 1.1] if config.get('speed_perturbation', False) else None,
                   window_size=config.get('window_size', .02),
                   window_stride=config.get('window_stride', .01),
                   nfilt=config.get('features', 80),
                   n_fft=config.get('n_fft', 512),
                   frame_splicing=config.get('frame_splicing', 3),
                   dither=config.get('dither', constant),
                   preemph=.97,
                   pad_to=8,  # TBD, config.get('pad_to', 8),
                   audio_fp16_input=audio_fp16_input,
                   max_duration=config.get('max_duration', 16.7))

    @staticmethod
    def _dali_init_log(args: dict):
        if (not torch.distributed.is_initialized() or (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)):  # print once
            max_len = max([len(ii) for ii in args.keys()])
            fmt_string = '\t%' + str(max_len) + 's : %s'
            print('Initializing DALI with parameters:')
            for keyPair in sorted(args.items()):
                print(fmt_string % keyPair)

    @staticmethod
    def _div_ceil(dividend, divisor):
        return (dividend + (divisor - 1)) // divisor

    def _get_audio_len(self, inp):
        return self._div_ceil(self.get_shape(inp), self.frame_splicing)

    def _splice_frames(self, inp):
        """
        Frame splicing is implemented by transposing the input, padding it,
        reshaping and then transposing back.
        """
        out = self.splicing_transpose(inp)
        out = self.splicing_pad(out)
        out = self.splicing_reshape(out)
        return out

    def define_graph(self):
        audio = self.extsrc()
        if self.audio_fp16_input:
            audio = self.to_float32(audio)
        audio = self.preemph(audio)
        audio = self.spectrogram(audio)
        audio = audio + self.dither ** 2
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)
        audio_len = self._get_audio_len(audio)

        if self.frame_splicing > 1:
            audio = self._splice_frames(audio)

        audio = self.normalize(audio)
        audio = self.pad(audio)
        audio = self.to_float16(audio)
        return audio, audio_len
