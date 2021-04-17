#  OpenVINO Int8 Quantization for BERT

To generate the Int8 quantized OpenVINO BERT model, we followed the
instructions [here](https://docs.openvinotoolkit.org/latest/omz_demos_python_demos_bert_question_answering_demo_README.html) with modifications. See below.

#  Instructions
1. Download the Int8 ONNX reference model from the MLCommons Inference
[repository](https://github.com/mlcommons/inference/tree/master/language/bert).
2. Convert the input data type to int32:
```
python3 modify.py
```

3. Run the OpenVINO Model Optimizer:
```
python3 mo.py \
    -m <path_to_model>/bert_large_v1_1_fake_quant_int32.onnx \
    --input "input_ids,attention_mask,token_type_ids" \ 
    --input_shape "[1,384],[1,384],[1,384]" \
    --keep_shape_ops \
    --output 6703 \
```
Note that the reference model performs an unnecessary split of the start_logits and end_logits into
separate outputs, which we skip by specifying the pre-split tensor as the model output.

