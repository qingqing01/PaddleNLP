# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial
import argparse
import os
import sys
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset, load_dataset
from paddlenlp.utils.log import logger

from base_model import SemanticIndexBase
from data import convert_example, create_dataloader
from data import gen_id2corpus
from tqdm import tqdm

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=None, type=int, help="output_embedding_size")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()


if __name__ == "__main__":
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
        "ernie-1.0")

    model = SemanticIndexBase(
        pretrained_model, output_emb_size=args.output_emb_size)
    model = paddle.DataParallel(model)

    # Load pretrained semantic model
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        logger.info("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    id2corpus = gen_id2corpus(args.corpus_file)

    # conver_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    # Need better way to get inner model of DataParallel
    inner_model = model._layers


    all_embeddings = []

    for text_embeddings in tqdm(inner_model.get_semantic_embedding(corpus_data_loader)):
        all_embeddings.append(text_embeddings.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    # print(all_embeddings.shape)
    np.save('corpus_embedding',all_embeddings)
