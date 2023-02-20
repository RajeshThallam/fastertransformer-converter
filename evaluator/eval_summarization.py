# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import functools
import argparse
import os
import numpy as np
import evaluate
from transformers import T5Config
from tqdm import tqdm
import configparser
import datetime
import seqio
import t5
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import gcsfs
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_inference_server_client(url, concurrency, verbose):
    return httpclient.InferenceServerClient(
        url=url, concurrency=concurrency, verbose=verbose)


def prepare_tensor(name, input):
    t = httpclient.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_location', type=str,
                        help='Path to config.init file on GCS')
    parser.add_argument('--output_metrics', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--samples_path', type=str)
    parser.add_argument('--max_seq_len', type=int, default=200)

    args = parser.parse_args()
    args_dict = vars(args)

    model_config_location = args.model_config_location
    samples_path = args.samples_path
    output_len = args.max_seq_len
    top_k = 2

    fs = gcsfs.GCSFileSystem()
    
    # seqio tokenization
    VOCAB = t5.data.get_default_vocabulary()
    DEFAULT_OUTPUT_FEATURES = {
        "inputs": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
            required=False),
        "targets": seqio.Feature(
            vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
    }

    DEFAULT_PREPROCESSORS = [
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ]

    # Task definition
    task = seqio.Task(
        'xsum',
        source=seqio.TextLineDataSource(
            split_to_filepattern={'test': samples_path},
            skip_header_lines=0
        ),
        preprocessors=[
            functools.partial(
                t5.data.preprocessors.preprocess_tsv,
                field_delim="\t",
                num_fields=2,
                inputs_format="summarize: {0}",
                targets_format="{1}"),
            *DEFAULT_PREPROCESSORS,
        ],
        metric_fns=[t5.evaluation.metrics.rouge],
        output_features=DEFAULT_OUTPUT_FEATURES)

    dataset = task.get_dataset(
        sequence_length={'my':512}, 
        split='test').as_numpy_iterator()

    ckpt_config = configparser.ConfigParser()

    print(f"Copying and Reading config file from {model_config_location}")
    if fs.exists(model_config_location):
        fs.get(model_config_location, 'config.ini')
        ckpt_config.read_file(open('config.ini'))
    else:
        assert False, "[ERROR] This example only support loading model with FT format directly."

    encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                              d_model=ckpt_config.getint(
                                  "encoder", "d_model"),
                              d_kv=ckpt_config.getint("encoder", "d_kv"),
                              d_ff=ckpt_config.getint("encoder", "d_ff"),
                              num_layers=ckpt_config.getint(
                                  "encoder", "num_layers"),
                              num_decoder_layers=ckpt_config.getint(
                                  "encoder", "num_decoder_layers"),
                              num_heads=ckpt_config.getint(
                                  "encoder", "num_heads"),
                              relative_attention_num_buckets=ckpt_config.getint(
                                  "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                              feed_forward_proj=ckpt_config.get(
                                  "encoder", "feed_forward_proj"),
                              pad_token_id=ckpt_config.getint(
                                  "encoder", "pad_token_id"),
                              eos_token_id=ckpt_config.getint(
                                  "encoder", "eos_token_id"),
                              is_gated_act=ckpt_config.getboolean(
                                  "encoder", "is_gated_act", fallback=0),
                              )
    decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                              d_model=ckpt_config.getint(
                                  "decoder", "d_model"),
                              d_kv=ckpt_config.getint("decoder", "d_kv"),
                              d_ff=ckpt_config.getint("decoder", "d_ff"),
                              num_layers=ckpt_config.getint(
                                  "decoder", "num_layers"),
                              num_decoder_layers=ckpt_config.getint(
                                  "decoder", "num_decoder_layers"),
                              num_heads=ckpt_config.getint(
                                  "decoder", "num_heads"),
                              relative_attention_num_buckets=ckpt_config.getint(
                                  "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                              feed_forward_proj=ckpt_config.get(
                                  "decoder", "feed_forward_proj"),
                              pad_token_id=ckpt_config.getint(
                                  "decoder", "pad_token_id"),
                              eos_token_id=ckpt_config.getint(
                                  "decoder", "eos_token_id"),
                              decoder_start_token_id=ckpt_config.getint(
                                  "decoder", "decoder_start_token_id"),
                              is_gated_act=ckpt_config.getboolean(
                                  "decoder", "is_gated_act", fallback=0),
                              )

    def summarize_ft(datapoint, model_name, url, verbose):
        line_tokens = datapoint['inputs']
        request_parallelism = 10
        with create_inference_server_client(
            url,
            concurrency=request_parallelism,
            verbose=verbose) as client:
            input_token = line_tokens

            input_ids = np.array([input_token]).astype(np.uint32)
            mem_seq_len = np.array([input_token.size]).astype(np.uint32)
            mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])

            runtime_top_k = (
                top_k * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            runtime_top_p = 0.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            random_seed = 0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint64)
            is_return_log_probs = True * \
                np.ones([input_ids.shape[0], 1]).astype(bool)
            max_output_len = (
                output_len * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            bad_words_ids = np.array(
                [[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            stop_words_ids = np.array(
                [[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            beam_width = (
                1 * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            start_ids = decoder_config.decoder_start_token_id * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            end_ids = encoder_config.eos_token_id * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint32)

            inputs = [
                prepare_tensor("input_ids", input_ids),
                prepare_tensor("sequence_length", mem_seq_len),
                prepare_tensor("runtime_top_k", runtime_top_k),
                prepare_tensor("runtime_top_p", runtime_top_p),
                prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
                prepare_tensor("temperature", temperature),
                prepare_tensor("len_penalty", len_penalty),
                prepare_tensor("repetition_penalty", repetition_penalty),
                prepare_tensor("random_seed", random_seed),
                prepare_tensor("is_return_log_probs", is_return_log_probs),
                prepare_tensor("max_output_len", max_output_len),
                prepare_tensor("beam_width", beam_width),
                prepare_tensor("start_id", start_ids),
                prepare_tensor("end_id", end_ids),
                prepare_tensor("bad_words_list", bad_words_ids),
                prepare_tensor("stop_words_list", stop_words_ids),
            ]

            result = client.infer(model_name, inputs)
            output = result.as_numpy("output_ids")
            ft_output_len = result.as_numpy("sequence_length")

        tokens = output[0][0]
        output_lines = VOCAB.decode(output[0][0][:ft_output_len[0][0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    metric_ft = evaluate.load('rouge')
    model_predictions = []
    model_references = []

    ft_time = 0.0
    for idx, datapoint in enumerate(tqdm(dataset)):
        try:
            start_time = datetime.datetime.now()
            summary_ft, _ = summarize_ft(
                datapoint, 
                args_dict['model_name'], 
                args_dict['url'] , 
                args_dict['verbose'])
            stop_time = datetime.datetime.now()
            ft_time += (stop_time - start_time).total_seconds()
            model_predictions.append(summary_ft)
            model_references.append(
                datapoint['targets_pretokenized'].decode())
        except Exception as e:
            print(e)
            print('Error with datapoint : ', idx)

    computed_metrics_ft = metric_ft.compute(
        predictions=model_predictions,
        references=model_references
    )

    # Write results to GCS
    with fs.open(args_dict['output_metrics'], 'w') as fp:
        fp.write(json.dumps(computed_metrics_ft))


if __name__ == '__main__':
    main()
