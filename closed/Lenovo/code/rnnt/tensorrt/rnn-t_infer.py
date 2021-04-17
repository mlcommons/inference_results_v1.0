#!/usr/bin/env python3
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

import tensorrt as trt
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common import logging

import argparse
import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time


# Support

def get_dtype_info(input_dtype_str):
    datatype2size = {"fp32": 4, "fp16": 2, "int8": 1}
    datatype2ndtype = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}
    dtype_esize = datatype2size[input_dtype_str]
    infer_ndtype = datatype2ndtype[input_dtype_str]
    return (infer_ndtype, dtype_esize)


# Main classes

class HostDeviceMem(object):
    def __init__(self, host, device):
        self.host = host
        self.device = device


class EngineRunner():

    def __init__(self, engine_file, args, verbose=False, plugins=None):
        self.args = args
        self.engine_file = engine_file
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        if not os.path.exists(engine_file):
            raise ValueError("File {:} does not exist".format(engine_file))

        trt.init_libnvinfer_plugins(self.logger, "")
        if plugins is not None:
            for plugin in plugins:
                ctypes.CDLL(plugin)
        self.engine = self.load_engine(engine_file)

        self.d_inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    def load_engine(self, src_path):
        with open(src_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        return engine

    def debug_input_info(self, dev_input_l, input_data_l, header_str="debug_input_info"):
        for (d_input, inp) in zip(dev_input_l, input_data_l):
            logging.info("{:}: d_input = {}".format(header_str, d_input))
            logging.info("{:}: inp shape = {}".format(header_str, inp.shape))
            logging.info("{:}: {}".format(header_str, inp))

    def _run_encoder(self, inputs, batch_size=1):

        if self.args.debug_mode:
            self.debug_input_info(self.d_inputs, inputs, "_run_encoder::input")

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(d_input, inp, self.stream) for (d_input, inp) in zip(self.d_inputs, inputs)]

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for i in range(len(inputs)):
                input_shape = self.context.get_binding_shape(i)
                input_shape[0] = batch_size
                self.context.set_binding_shape(i, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        if self.args.debug_mode:
            # Synchronize the stream
            self.stream.synchronize()
            logging.info("out {:}".format(self.outputs[0].host))
            # self.debug_input_info([self.outputs[0].device], [self.outputs[0].host], "_run_encoder::output")

        # Return only the host outputs.
        # logging.info("run_encoder::actual_batch_size = {}".format(batch_size))
        # logging.info("run_encoder::output tensor shape = {} ntype = {}".format(self.outputs[0].host.shape, self.outputs[0].host.dtype))
        return [out.host for out in self.outputs]

    def _run_decoder(self, inputs, seq_id, batch_size=1):
        (infer_ndtype, dtype_esize) = get_dtype_info(self.args.input_dtype)

        # self.debug_input_info(self.d_inputs, inputs, "_run_decoder")

        # Transfer input data to the GPU
        if seq_id == 0:
            # iter 0 needs the initial state
            hidden_tensor = np.ascontiguousarray(np.zeros((batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C'))     # layers * hidden
            cell_tensor = np.ascontiguousarray(np.zeros((batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C'))     # layers * hidden
            [cuda.memcpy_htod_async(d_input, inp, self.stream) for (d_input, inp) in zip(self.d_inputs, [inputs[0], hidden_tensor, cell_tensor])]
        else:
            # rest of iteration auto-reccur the state
            cuda.memcpy_htod_async(self.d_inputs[0], inputs[0], self.stream)

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for inp_idx in range(3):
                input_shape = self.context.get_binding_shape(inp_idx)
                input_shape[0] = batch_size
                self.context.set_binding_shape(inp_idx, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        if self.args.debug_mode:
            # Transfer all outputs back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            # Synchronize the stream
            self.stream.synchronize()
            logging.info("_run_decoder: out[0] = {}".format(self.outputs[0].host))
            self.debug_input_info(self.d_inputs, [out.host for out in self.outputs], "_run_decoder")

        # [cuda.memcpy_dtod_async(d_input, out.device, self.stream) for (d_input, out) in zip(self.d_inputs, self.outputs)]
        hidden_size = self.hyperP.decoder_hidden_size
        input_size = self.hyperP.decoder_input_size

        # Update state for next iteration
        cuda.memcpy_dtod_async(self.d_inputs[1], self.outputs[1].device, batch_size * 2 * hidden_size * dtype_esize, self.stream)
        cuda.memcpy_dtod_async(self.d_inputs[2], self.outputs[2].device, batch_size * 2 * hidden_size * dtype_esize, self.stream)

        # Transfer output to host
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        # logging.info("_run_decoder: out[0] = {}".format(self.outputs[0].host))

        # Synchronize the stream
        self.stream.synchronize()

        # return the 'symbol' host output
        return self.outputs[0].host

    def decoder_step(self, batch_size=1):

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for inp_idx in range(3):
                input_shape = self.context.get_binding_shape(inp_idx)
                input_shape[0] = batch_size
                self.context.set_binding_shape(inp_idx, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # return list of device outputs
        return [out.device for out in self.outputs]

    def _run_joint(self, inputs, batch_size=1):

        # self.debug_input_info(self.d_inputs, inputs, "_run_joint")

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(d_input, inp, self.stream) for (d_input, inp) in zip(self.d_inputs, inputs)]
        # [ logging.info(" inp: {:} -- {:}".format(d_input, inp)) for (d_input, inp) in zip(self.d_inputs, inputs)]

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for inp_idx in range(2):
                input_shape = self.context.get_binding_shape(inp_idx)
                input_shape[0] = batch_size
                self.context.set_binding_shape(inp_idx, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        if self.args.debug_mode:
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

            # Synchronize the stream
            self.stream.synchronize()
            logging.info("out {:}".format(self.outputs[0].host))

        # return the 'symbol' host output
        return [self.outputs[0].host]

    def joint_step(self, inputs, batch_size=1):

        # Run inference.
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        else:
            for inp_idx in range(2):
                input_shape = self.context.get_binding_shape(inp_idx)
                input_shape[0] = batch_size
                self.context.set_binding_shape(inp_idx, input_shape)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

    def __call__(self, inputs, batch_size=1):
        if self.engine_name == "encoder":
            return self._run_encoder(inputs, batch_size)
        elif self.engine_name == "decoder":
            return self._run_decoder(inputs, batch_size)
        elif self.engine_name == "joint":
            return self._run_joint(inputs, batch_size)
        else:
            raise(Exception("Invalid topology: {}".format(self.args.topology)))

    def __del__(self):
        # Clean up everything.
        with self.engine, self.context:
            [d_input.free() for d_input in self.d_inputs]
            [out.device.free() for out in self.outputs]
            del self.stream

    def allocate_buffers(self):
        d_inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        # FIXME: Clarify this with Po-han. What if we want a batch size smaller than max_batch_size from builder?
        # max_batch_size = self.engine.max_batch_size if self.engine.has_implicit_batch_dimension else self.engine.get_profile_shape(0, 0)[2][0]
        if self.engine.has_implicit_batch_dimension:
            max_batch_size = min(self.engine.max_batch_size, self.args.batch_size)
        else:
            max_batch_size = self.args.batch_size

        for binding in self.engine:
            logging.info("Binding {:}".format(binding))
            desc = self.engine.get_binding_format_desc(self.engine.get_binding_index(binding))
            logging.info("    Binding info {:} with shape {:}".format(desc, self.engine.get_binding_shape(self.engine.get_binding_index(binding))))
            dtype = self.engine.get_binding_dtype(binding)
            format = self.engine.get_binding_format(self.engine.get_binding_index(binding))
            shape = self.engine.get_binding_shape(binding)
            if format == trt.TensorFormat.CHW4:
                shape[-3] = ((shape[-3] - 1) // 4 + 1) * 4
            if not self.engine.has_implicit_batch_dimension:
                shape[0] = max_batch_size
                size = trt.volume(shape)
            else:
                size = trt.volume(shape) * max_batch_size
            # Allocate host and device buffers
            device_mem = cuda.mem_alloc(size * dtype.itemsize)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                d_inputs.append(device_mem)
            else:
                host_mem = cuda.pagelocked_empty(size, trt.nptype(dtype))
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return d_inputs, outputs, bindings, stream


class EncoderRunner(EngineRunner):
    def __init__(self, args, _hyperP):
        self.engine_name = 'encoder'
        self.hyperP = _hyperP
        super().__init__(os.path.join(args.engine_dir, "encoder.plan"), args, True)


class DecoderRunner(EngineRunner):
    def __init__(self, args, _hyperP):
        self.engine_name = 'decoder'
        self.hyperP = _hyperP
        super().__init__(os.path.join(args.engine_dir, "decoder.plan"), args, True)


class JointRunner(EngineRunner):
    def __init__(self, args, _hyperP):
        self.engine_name = 'joint'
        self.hyperP = _hyperP
        super().__init__(os.path.join(args.engine_dir, "joint.plan"), args, True)


class RNNTRunner(object):
    def __init__(self, args, _hyperP):
        self.args = args
        self.hyperP = _hyperP
        self.engine_dir = args.engine_dir
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.data_dir = args.data_dir
        self.length_dir = args.length_dir
        self.val_map = args.val_map
        self.load_val_images()
        self.batch_inputs = np.ascontiguousarray(np.stack([np.load(os.path.join(self.data_dir, name + ".npy")) for name in self.input_list[:self.batch_size]]))
        self.batch_lengths = np.ascontiguousarray(np.stack([np.load(os.path.join(self.length_dir, name + ".npy")) for name in self.input_list[:self.batch_size]]))
        if self.args.topology == "encoder":
            self.encoder = EncoderRunner(args, _hyperP)
        elif self.args.topology == "decoder":
            self.decoder = DecoderRunner(args, _hyperP)
        elif self.args.topology == "joint":
            self.joint = JointRunner(args, _hyperP)
        elif self.args.topology == "greedy":
            self.encoder = EncoderRunner(args, _hyperP)
            self.decoder = DecoderRunner(args, _hyperP)
            self.joint = JointRunner(args, _hyperP)
        else:
            raise(Exception("Invalid topology: {}".format(self.args.topology)))

    def load_val_images(self):
        self.input_list = []
        with open(self.val_map) as f:
            for line in f:
                self.input_list.append(line.split()[0])

    # Encoder runner
    def infer_encoder(self):
        batch_idx = 0
        self.outputs = []
        for image_idx in range(0, self.num_samples, self.batch_size):
            # Actual batch size might be smaller than max batch size
            actual_batch_size = self.batch_size if image_idx + self.batch_size <= self.num_samples else self.num_samples - image_idx

            start_time = time.time()
            outputs = self.encoder([self.batch_inputs[:actual_batch_size], self.batch_lengths[:actual_batch_size]], actual_batch_size)
            self.outputs.extend(outputs[0][:actual_batch_size])

            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))
            batch_idx += 1

    # Decoder runner
    def infer_decoder(self):
        batch_idx = 0
        max_seq_length = 1152 // 2

        (infer_ndtype, dtype_esize) = get_dtype_info(self.args.input_dtype)

        self.outputs = []
        for image_idx in range(0, self.num_samples, self.batch_size):
            # Actual batch size might be smaller than max batch size
            actual_batch_size = self.batch_size if image_idx + self.batch_size <= self.num_samples else self.num_samples - image_idx

            start_time = time.time()

            input_port = np.ascontiguousarray(np.random.randint(0, high=self.hyperP.labels_size, size=(actual_batch_size, 1)), dtype=np.int32)

            # iterate over seq id
            for seq_id in range(max_seq_length):
                predictions = self.decoder._run_decoder([input_port], seq_id, actual_batch_size)
                predictions = predictions.reshape((actual_batch_size, self.hyperP.decoder_hidden_size))
                winners = np.argmax(predictions, axis=1)
                self.outputs.extend(winners[:actual_batch_size])
                input_port = np.minimum(winners, self.hyperP.labels_size - 1)
                input_port = np.ascontiguousarray(input_port.reshape((actual_batch_size, 1)), dtype=np.int32)

            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))
            batch_idx += 1

    # Joint runner
    def infer_joint(self):
        (infer_ndtype, dtype_esize) = get_dtype_info(self.args.input_dtype)

        batch_idx = 0
        self.outputs = []
        for image_idx in range(0, self.num_samples, self.batch_size):
            # Actual batch size might be smaller than max batch size
            actual_batch_size = self.batch_size if image_idx + self.batch_size <= self.num_samples else self.num_samples - image_idx

            start_time = time.time()

            max_seq_length = 1152 + 1152 // 2   # U=1152//2 + T=1152   FIXME
            encoder_input_size = self.hyperP.encoder_hidden_size
            decoder_input_size = self.hyperP.decoder_hidden_size

            for seq_idx in range(max_seq_length):
                # input ports
                enc_input_port = np.ascontiguousarray(np.random.rand(actual_batch_size, 1, encoder_input_size), dtype=infer_ndtype)
                dec_input_port = np.ascontiguousarray(np.random.rand(actual_batch_size, 1, decoder_input_size), dtype=infer_ndtype)
                inputs = [enc_input_port, dec_input_port]

                outputs = self.joint(inputs, actual_batch_size)
                self.outputs.extend(outputs[0][:actual_batch_size])

            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))
            batch_idx += 1

    # Greedy runner
    def infer_greedy(self):
        (infer_ndtype, dtype_esize) = get_dtype_info(self.args.input_dtype)
        batch_idx = 0

        # Static knobs
        ALWAYS_ADVANCE_TIME = True    # hack to always consume time pointer regardless of symbol outcome

        # Iterate over batches
        for image_idx in range(0, self.num_samples, self.batch_size):
            # Actual batch size might be smaller than max batch size
            actual_batch_size = self.batch_size if image_idx + self.batch_size <= self.num_samples else self.num_samples - image_idx

            start_time = time.time()

            # output and runtime data structures
            #
            enc_ptr = [0 for tdix in range(actual_batch_size)]         # holds encoder pointer per batch element (Xt)
            out_sym = [list() for tdix in range(actual_batch_size)]    # holds output symbol translation per batch element (Yu)

            # data initialization for the batch ----------
            #

            # dec_inputs : host data for the decoder transfers
            dec_host_inputs = [
                np.ascontiguousarray(np.zeros((actual_batch_size, 1), dtype=np.int32, order='C')),                                        # input label
                np.ascontiguousarray(np.zeros((actual_batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C')),     # hiden: layers * hidden
                np.ascontiguousarray(np.zeros((actual_batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C')),     # cell:  layers * hidden
            ]

            # host_outputs : host data for outputs from decoder and joint/beam_search
            host_outputs = [
                np.ascontiguousarray(np.zeros((actual_batch_size, 1 * self.hyperP.labels_size), dtype=infer_ndtype, order='C')),             # input: 1 * input
                np.ascontiguousarray(np.zeros((actual_batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C')),     # hiden: layers * hidden
                np.ascontiguousarray(np.zeros((actual_batch_size, 2 * self.hyperP.decoder_hidden_size), dtype=infer_ndtype, order='C')),     # cell:  layers * hidden
            ]

            # run the encoder ----------
            #

            #  outputs[0] - ( BS, max_seq_length // 2, enc_hidden_size=1024 )
            enc_outputs = self.encoder([self.batch_inputs[:actual_batch_size]], actual_batch_size)
            self.encoder.stream.synchronize()
            enc_outputs[0] = enc_outputs[0].reshape((actual_batch_size, self.hyperP.max_seq_length // 2, self.hyperP.encoder_hidden_size))
            # logging.info(" greedy::enc_output shape {:} type {:}".format(enc_outputs[0].shape, enc_outputs[0].dtype ))
            # logging.info(" greedy::enc_output data\n{:}".format(enc_outputs[0]))

            # run the decoder-joint greedy loop ----------
            #

            for seq_id in range(self.hyperP.max_seq_length // 2):
                # enc_input_seq = np.ascontiguousarray(enc_outputs[0][:,seq_id,:])
                # enc_input_seq = enc_input_seq.reshape (actual_batch_size, 1,  self.hyperP.encoder_hidden_size)
                enc_input_seq = np.ascontiguousarray(np.zeros((actual_batch_size, 1, self.hyperP.encoder_hidden_size), dtype=infer_ndtype, order='C'))
                for bs_index in range(actual_batch_size):
                    enc_input_seq[bs_index, :, :] = enc_outputs[0][bs_index, enc_ptr[seq_id], :]
                # logging.info(" greedy::enc_output seq[{}] shape {:} data\n{:}".format(seq_id, enc_input_seq.shape, enc_input_seq))

                # run decoder/predictor (transfer data first)
                [cuda.memcpy_htod_async(d_input, inp, self.decoder.stream) for (d_input, inp) in zip(self.decoder.d_inputs, dec_host_inputs)]
                # self.debug_input_info(self.d_inputs, inputs, "_run_decoder")
                dec_dev_outputs = self.decoder.decoder_step(actual_batch_size)

                # trasfer decoding state to host
                cuda.memcpy_dtoh_async(host_outputs[1], dec_dev_outputs[1], self.decoder.stream)
                cuda.memcpy_dtoh_async(host_outputs[2], dec_dev_outputs[2], self.decoder.stream)

                # transfer data for joint
                # logging.info("tensor: shape = {:} -- {} :\n{}".format(enc_input_seq.shape, enc_input_seq.dtype, enc_input_seq))
                cuda.memcpy_htod_async(self.joint.d_inputs[0], enc_input_seq, self.joint.stream)  # encoder port
                self.decoder.stream.synchronize()
                cuda.memcpy_dtod_async(self.joint.d_inputs[1], dec_dev_outputs[0], actual_batch_size * self.hyperP.decoder_hidden_size * dtype_esize, self.joint.stream)  # predictor

                # run joint
                self.joint.joint_step(actual_batch_size)

                # Transfer result to CPU for greedy decoder
                cuda.memcpy_dtoh_async(host_outputs[0], self.joint.outputs[0].device, self.joint.stream)
                self.joint.stream.synchronize()

                # greedy decoder
                winner_symbol = np.argmax(host_outputs[0], axis=1)
                # logging.info("Joint outputs:\n{} ".format(host_outputs[0]))
                # logging.info("Winner_symbol:\n{} ".format(winner_symbol))
                for bs_index in range(actual_batch_size):
                    new_symbol = winner_symbol[bs_index]
                    if new_symbol != self.hyperP.labels_size - 1:
                        # symbol is not blank
                        dec_host_inputs[0][bs_index, 0] = winner_symbol[bs_index]       # update predicted symbol
                        dec_host_inputs[1][bs_index, :] = host_outputs[1][bs_index, :]  # update hidden state
                        dec_host_inputs[2][bs_index, :] = host_outputs[2][bs_index, :]  # update cell state
                        out_sym[bs_index].append(winner_symbol[bs_index])
                        if ALWAYS_ADVANCE_TIME:
                            enc_ptr[bs_index] += 1
                        # logging.info("Adding symbol {} in bs_id {} ".format(winner_symbol[bs_index], bs_index))
                    else:
                        # advance the audio time pointer if the symbol is blank
                        enc_ptr[bs_index] += 1

            # Loop epilogue
            logging.info("Batch {:d} (Size {:}) >> Inference time: {:f}".format(batch_idx, actual_batch_size, time.time() - start_time))
            batch_idx += 1
            # logging.info("   output sequences:\n{:}".format(out_sym))

        # Function epilogue
        pass

    # Infer wrdapper
    def infer(self):
        if self.args.topology == "encoder":
            self.infer_encoder()
        elif self.args.topology == "decoder":
            self.infer_decoder()
        elif self.args.topology == "joint":
            self.infer_joint()
        elif self.args.topology == "greedy":
            self.infer_greedy()
        else:
            raise(Exception("Invalid topology: {}".format(self.args.topology)))


# Common parameters
##

class RnnHyperParam():
    def __init__(self, args):
        # Batch and sequence length
        self.batch_size = args.batch_size
        self.max_seq_length = 1152

        # alphabet
        self.labels_size = 29   # alphabet

        # encoder
        self.encoder_input_size = 240
        self.encoder_hidden_size = 1024
        self.enc_pre_rnn_layers = 2
        self.enc_post_rnn_layers = 3

        # encoder
        self.decoder_input_size = 512
        self.decoder_hidden_size = 512
        self.dec_rnn_layers = 2


def infer(args):
    hyperParam = RnnHyperParam(args)
    runner = RNNTRunner(args, hyperParam)
    logging.info("Start running inference -- topology : {:}".format(args.topology))
    start = time.time()
    runner.infer()
    end = time.time()
    elapsed = end - start
    logging.info("Inference takes {:f} secs. Throughput = {:f}/s".format(elapsed, args.num_samples / elapsed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=12800)
    parser.add_argument("--debug_mode", default=False, action='store_true', help='Debug mode to inspect data')
    parser.add_argument("--engine_dir", default="build/engines/rnnt")
    parser.add_argument("--val_map", default="data_maps/rnnt_1152/val_map.txt")
    parser.add_argument("--data_dir", default="build/preprocessed_data/rnnt_1152/fp16")
    parser.add_argument("--length_dir", default="build/preprocessed_data/rnnt_1152/int32")
    parser.add_argument("--input_dtype", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--topology", default="encoder", help="Options: encoder/decoder/joint/greedy")
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
