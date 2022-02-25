# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
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
import numpy as np
 
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.tensor as tensor
from paddle.nn import Layer, Embedding
 
from .. import PretrainedModel, register_base_model
 
__all__ = [
]
 
class ErnieViLGPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Bart models. It provides Bart related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "vilg-10b": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 4096,
            "intermediate_hidden_size": 16384,
            "out_emb_size": 4096,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "max_img_position_embeddings": 1026,
            "max_img_row": 34,
            "num_attention_heads": 64,
            "num_hidden_layers": 48,
            "type_vocab_size": 4,
            "vocab_size": 19000,
            "img_vocab_size":8194
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {}
    }
    base_model_prefix = "vilg"
 
    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.init_std if hasattr(self, "init_std") else
                        self.bart.config["init_std"],
                        shape=layer.weight.shape))
 
 
class ViLGEmbedding(nn.Layer):
    """
    Text and image embedding.
    """
    def __init__(self,
                 vocab_size,
                 emb_size,
                 max_position_seq_len,
                 img_voc_size,
                 max_img_row,
                 tied_embedding=True,
                 initializer_range=0.02,
                 name=""):
        super(ViLGEmbeddings, self).__init__()
 
        self._model_name = name
        self._rel_pos_emb_name = self._model_name + "rel_pos_embedding"
        if tied_embedding:
            self._word_emb_name = "word_embedding"
            self._pos_emb_name = "pos_embedding"
            self._sent_emb_name = "sent_embedding"
        else:
            self._word_emb_name = self._model_name + "word_embedding"
            self._img_emb_name = self._model_name + "img_embedding"
            self._pos_emb_name = self._model_name + "pos_embedding"
            self._img_pos_emb_name = self._model_name + "img_pos_embedding"
            self._sent_emb_name = self._model_name + "sent_embedding"
        
        self._pb_relax = False 
        self._ln_type = ln_type
        if ln_type == "post":
            self._preprocess_cmd = ""
            self._postprocess_cmd = "dan"
        elif ln_type == "pre":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "da"
        elif ln_type == "pre-post":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "dan"
        elif ln_type == "sandwich":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "dna"
            self._pb_relax = True 
 
        self._param_initializer = nn.initializer.TruncatedNormal(
            scale=initializer_range)
 
        self.word_emb = nn.Embedding(vocab_size, emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._word_emb_name,
                initializer=self._param_initializer))
 
        self.position_emb = nn.Embedding(max_position_seq_len, emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._pos_emb_name,
                initializer=self._param_initializer))
 
        self.img_emb = nn.Embedding(img_voc_size, emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._img_emb_name,
                initializer=self._param_initializer))
 
        self.img_row_position_emb = nn.Embedding(max_img_row, emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._img_pos_emb_name + "_row",
                initializer=self._param_initializer))
 
        self.img_col_position_emb = nn.Embedding(max_img_row, emb_size,
            weight_attr=paddle.ParamAttr(
                name=self._img_pos_emb_name + "_col",
                initializer=self._param_initializer))
 
    def forward(self, src_ids, position_ids, is_encoding, is_decoding):
        """
        src_ids: list size is 2
        pos_ids: list size is 3
        """
        emb_out = self.word_emb(src_ids[0])
        position_emb_out = self.position_emb(position_ids[0])
        emb_out_txt = emb_out + position_emb_out
 
        img_emb_out = self.img_emb(src_ids[1])
        img_row_position_emb_out = self.img_row_position_emb(position_ids[1])
        img_col_position_emb_out = self.img_col_position_emb(position_ids[2])
 
        emb_out_img = img_emb_out + img_row_position_emb_out + img_col_position_emb_out
        emb_out = paddle.multiply(emb_out_txt, is_encoding) + paddle.multiply(emb_out_img, is_decoding)
 
        return emb_out
 
 
class PrePostProcessLayer(nn.Layer):
    """
    Add residual connection, layer normalization and droput to the out tensor
    optionally according to the value of process_cmd.
    This will be used before or after multi-head attention and position-wise
    feed-forward networks.
    """
    def __init__(self,
                 process_cmd,
                 dropout_rate=0.,
                 hidden_size=0.,
                 name=''):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.dropout_rate = dropout_rate
        self.norm = None
        if 'n' in process_cmd:
            self.norm = nn.LayerNorm(hidden_size,
                weight_attr=paddle.ParamAttr(
                    name=name + '_layer_norm_scale',
                    initializer=nn.initializer.Constant(1.)),
                bias_attr=paddle.ParamAttr(
                    name=name + '_layer_norm_bias',
                    initializer=nn.initializer.Constant(0.)))
        self.dropout = None
        if 'd' in process_cmd and self.dropout_rate:
            self.dropout = nn.Dropout(self.dropout_rate, mode="upscale_in_train")
 
    def forward(self, pre_out, out):
        for cmd in self.process_cmd:
            if cmd == "a":  # add residual connection
                out = out + prev_out if prev_out else out
            elif cmd == "n":  # add layer normalization
                out = self.norm(out)
            elif cmd == "d":  # add dropout
                if self.dropout_rate:
                    out = self.dropout(out)
        return out
 
class ErnieViLGEncoderLayer(nn.Layer):
    """
    The Ernie Encoder Layer.
    """
    def __init__(self,
                  emb_size,
                  hidden_size,
                  num_heads,
                  d_key,
                  d_value,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  pb_relax=False,
                  initializer_range=0.02,
                  name=''):
        super().__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self._param_initializer = nn.initializer.TruncatedNormal(
            scale=initializer_range)
 
        self.pre_att = PrePostProcessLayer(preprocess_cmd,
                prepostprocess_dropout, hidden_size, name=name + '_pre_att')
        self.self_att = nn.MultiHeadAttention(hidden_size, num_heads,
                attention_dropout)
        self.post_att = PrePostProcessLayer(postprocess_cmd,
                prepostprocess_dropout, hidden_size, name=name + '_post_att')
        self.pre_ffn = PrePostProcessLayer(preprocess_cmd,
                prepostprocess_dropout, hidden_size, name=name + 'pre_ffn')
 
        self.linear1 = Linear(
            hidden_size, d_inner_hid,
            weight_attr=paddle.ParamAttr(
                name=name + '_fc_0.w_0', initializer=self._param_initializer),
            bias_attr=name + '_fc_0.b_0')
        self.dropout = None
        if relu_dropout:
            self.dropout = Dropout(relu_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            d_inner_hid, hidden_size,
            weight_attr=paddle.ParamAttr(
                name=name + '_fc_1.w_0', initializer=self._param_initializer),
            bias_attr=name + '_fc_1.b_0')
 
        self.post_ffn = PrePostProcessLayer(postprocess_cmd,
            prepostprocess_dropout, hidden_size, name=name + 'pre_ffn')
        self.activation = getattr(F, hidden_act)
 
 
        #self.apply(self.init_weights)
 
    def forward(self, enc_input, pos_bias, attn_bias):
        """
        """
        enc_ln_out = self.pre_att(enc_input)
        attn_out = self.self_att(enc_ln_out, attn_mask=attn_bias)
        attn_out = self.post_att(attn_out)
        attn_ln_out = self.pre_ffn(attn_out)
 
        ffd_out = self.linear1(attn_ln_out)
        ffd_out = self.activation(ffd_out)
        if self.dropout:
            ffd_out = self.dropout(ffd_out)
        ffd_out = self.linear2(ffd_out)
        out = self.post_ffn(ffd_out)
        return out
 
 
class ErnieViLGEncoder(nn.Layer):
    """
    The Ernie Encoder
    """
    def __init__(self,
                  num_layers,
                  emb_size,
                  hidden_size,
                  num_heads,
                  d_key,
                  d_value,
                  d_inner_hid,
                  prepostprocess_dropout,
                  attention_dropout,
                  relu_dropout,
                  hidden_act,
                  preprocess_cmd="n",
                  postprocess_cmd="da",
                  pb_relax=False,
                  initializer_range=0.02,
                  name=''):
                 num_layers):
        super(ErnieEncoder, self).__init__()
        self.layers = LayerList([
            ErnieViLGEncoderLayer(
                emb_size=emb_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                d_key=d_key,
                d_value=d_value,
                d_inner_hid=d_inner_hid,
                prepostprocess_dropou=prepostprocess_dropout,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                hidden_act=hidden_act,
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
                pb_relax=pb_relax,
                initializer_range=initializer_range,
                name=name) for i in range(num_layers)
        ])
 
        self.num_layers = num_layers
 
    def forward(self, enc_input, pos_bias, attn_bias, cache=None):
        for i, mod in enumerate(self.layers):
            if i == self.num_layers - 1:
                attn_id = 2
            elif (i - 1) % 4 == 0:
                attn_id = 1
            else:
                attn_id = 0
    
            cache = None
            if caches:
                cache = caches[i]
    
            enc_output = mod(enc_input, pos_bias, attn_bias[attn_id])
        return enc_output
 
 
@register_base_model
class ErnieViLGModel(ErnieViLGPretrainedModel):
    """
    """
    def __init__(
            self,
            config,
            rel_pos_bin=32,
            weight_sharing=True,
            tied_embedding=True,
            use_gumbel_softmax=False,
            ln_type='pre',
            name=''):
        super(ErnieViLGModel, self).__init__()
 
        self._hidden_size = config['hidden_size']
        self._intermediate_hidden_size = config['intermediate_hidden_size'] or (config['hidden_size'] * 4)
        self._emb_size = config['emb_size'] or self._hidden_size
        self._out_emb_size = config['out_emb_size'] or self._emb_size
        self._max_seq_len = config["max_seq_len"]
        
        self._voc_size = config['vocab_size']
        self._img_voc_size = config['img_vocab_size']
        self._rel_pos_bin = rel_pos_bin
        self._out_voc_size = config['out_vocab_size'] or self._voc_size
        
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._max_position_seq_len = config['max_position_embeddings']
        self._max_img_position_seq_len = config['max_img_position_embeddings']
        self._max_img_row = config['max_img_row']
        self._sent_types = config['type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing
        self._use_gumbel_softmax = use_gumbel_softmax
        
        self._model_name = name
        self._rel_pos_emb_name = self._model_name + "rel_pos_embedding"
        
        self._pb_relax = False 
        self._ln_type = ln_type
        if ln_type == "post":
            self._preprocess_cmd = ""
            self._postprocess_cmd = "dan"
        elif ln_type == "pre":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "da"
        elif ln_type == "pre-post":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "dan"
        elif ln_type == "sandwich":
            self._preprocess_cmd = "n"
            self._postprocess_cmd = "dna"
            self._pb_relax = True 
 
        self._dtype = "float32"
 
 
        self.embeddings = ViLGEmbeddings(
            vocab_size, hidden_size, max_position_seq_len,
            img_voc_size, max_img_row, name=name)
        self.pre_encoder = PrePostProcessLayer('nd',
                prepostprocess_dropout, hidden_size, name=name + '_pre_encoder')
 
        if self._emb_size != self._hidden_size:
            self.fc_mapping = nn.Linear()
            self.fc_mapping = Linear(
                self._emb_size, self._hidden_size,
                weight_attr=paddle.ParamAttr(
                    name=name + 'emb_hidden_mapping'),
                bias_attr=name + 'emb_hidden_mapping_bias')
 
        self.encoder = ErnieViLGEncoder(
            num_layers=num_layers,
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            d_key=d_key,
            d_value=d_value,
            d_inner_hid=d_inner_hid,
            prepostprocess_dropout=prepostprocess_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            hidden_act=hidden_act,
            preprocess_cmd=preprocess_cmd,
            postprocess_cmd=postprocess_cmd,
            pb_relax=pb_relax,
            initializer_range=initializer_range,
            name=name)
 
        self.post_encoder = PrePostProcessLayer(self._preprocess_cmd,
                self._prepostprocess_dropout, name=name+'post_encoder')
 
        #self.apply(self.init_weights)
 
    def forward(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 input_mask,
                 is_encoding,
                 is_decoding,
                 cache_idxes,
                 cache=None):
        """
 
        Args:
            src_ids: list size is 2
            position_ids: list size is 3
        """
        emb_out = self.embeddings(src_ids, position_ids, is_encoding, is_decoding)
        emb_out = self.pre_encoder(emb_out)
 
        if self._emb_size != self._hidden_size:
            emb_out = self.fc_mapping(emb_out)
 
        self_attn_mask = input_mask
        self_attn_mask = map(lambda x : paddle.scale(
            x=x, scale=10000.0, bias=-1.0, bias_after_scale=False), self_attn_mask)
        n_head_attn_mask = map(lambda x : paddle.stack(
            x=[x] * self._n_head, axis=1), self_attn_mask)
        n_head_attn_mask = list(n_head_self_attn_mask)
 
        for i in range(len(n_head_self_attn_mask)):
            n_head_self_attn_mask[i].stop_gradient = True
 
        enc_out = self.encoder(emb_out, pos_input=None, attn_bias=n_head_attn_mask)
        enc_out = self.post_encoder(enc_out)
        return enc_out