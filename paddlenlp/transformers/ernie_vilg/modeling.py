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
"""Modeling classes for UnifiedTransformer model."""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder
from paddle.nn.layer.transformer import _convert_param_attr_to_list

from .. import PretrainedModel, register_base_model

# __all__ = [
#     "UnifiedTransformerPretrainedModel",
#     'UnifiedTransformerModel',
#     'UnifiedTransformerLMHeadModel',
#     'UnifiedTransformerForMaskedLM',
# ]


class ErnieVilGPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained UnifiedTransformer models. It provides  UnifiedTransformer
    related `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading
    and loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "ernie-vilg-10b": {
            "vocab_size": 30001,
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "normalize_before": True,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "unk_token_id": 0,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "mask_token_id": 30000,
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "unified_transformer-12L-cn":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn.pdparams",
            "unified_transformer-12L-cn-luge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/unified_transformer-12L-cn-luge.pdparams",
            "plato-mini":
            "https://bj.bcebos.com/paddlenlp/models/transformers/unified_transformer/plato-mini.pdparams",
        }
    }
    base_model_prefix = "unified_transformer"

    def init_weights(self, layer):
        # Initialization hook
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.unified_transformer.config["initializer_range"],
                        shape=layer.weight.shape))


class ErnieVilGEmbeddings(nn.Layer):
    #Include embeddings from word, position and token_type.

    def __init__(self,
                 vocab_size,
                 img_vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=513,
                 max_img_row=34,
                 max_img_col=34,
                 ):
        super(ErnieVilGEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.img_embeddings = nn.Embedding(img_vocab_size,
                                                hidden_size)
        self.row_position_embeddings = nn.Embedding(max_img_row,
                                                hidden_size)
        self.col_position_embeddings = nn.Embedding(max_img_col,
                                                hidden_size)

        # self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, text_ids, img_ids, position_ids, img_row_pos_ids, img_col_pos_ids, is_encoding):
        input_embedings = self.word_embeddings(text_ids)
        position_embeddings = self.position_embeddings(position_ids)
        text_embeddings = input_embedings + position_embeddings

        img_embeddings = self.img_embeddings(img_ids)
        row_position_embeddings = self.row_position_embeddings(img_row_pos_ids)
        col_position_embeddings = self.col_position_embeddings(img_col_pos_ids)
        
        image_embeddings = img_embeddings + row_position_embeddings + col_position_embeddings

        if is_encoding:
            embeddings = text_embeddings
        else:
            embeddings = image_embeddings
        embeddings = self.ln(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErnieVilgEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(ErnieVilgEncoderLayer, self).__init__(d_model, 
                                                    nhead, 
                                                    dim_feedforward, 
                                                    dropout, 
                                                    activation,
                                                    attn_dropout,
                                                    act_dropout,
                                                    normalize_before,
                                                    weight_attr,
                                                    bias_attr)

        assert d_model > 0, ("Expected d_model to be greater than 0, "
                             "but recieved {}".format(d_model))
        assert nhead > 0, ("Expected nhead to be greater than 0, "
                           "but recieved {}".format(nhead))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but recieved {}".format(dim_feedforward))

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = nn.MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = nn.Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        # self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, src_mask=None, cache=None):
        r"""
        Applies a Transformer encoder layer on the input.

        Parameters:
            src (Tensor): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (Tensor, optional): It is an instance of `MultiHeadAttention.Cache`.
                See `TransformerEncoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `enc_input`, representing the output of Transformer encoder \
                layer. Or a tuple if `cache` is not None, except for encoder \
                layer output, the tuple includes the new cache which is same \
                as input `cache` argument but `incremental_cache` has an \
                incremental length. See `MultiHeadAttention.gen_cache` and \
                `MultiHeadAttention.forward` for more details.
        """
        # src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        
        # Add cache for encoder for the usage like UniLM
        if cache is None:
            src = self.self_attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.self_attn(src, src, src, src_mask,
                                                    cache)
        # src = residual + self.dropout1(src)
        src = self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        # print('src shape 01:', src.shape)
        # residual = src
        if self.normalize_before:
            src = self.norm2(src)
        if len(residual.shape) != len(src.shape):
            residual = residual.squeeze(1)
        src = residual + src
        residual = src
        # layer norm
        src = self.norm3(src)
        # ffn
        src = self.linear2(self.activation(self.linear1(src)))
        # src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = residual + self.dropout2(src)
        src = self.dropout2(src)
        src = residual + self.norm4(src)
        # print('src shape 03:', src.shape)
        # if not self.normalize_before:
        #     src = self.norm2(src)
        return src if cache is None else (src, incremental_cache)

    def gen_cache(self, k, v):
        # incremental_cache = self.self_attn.StaticCache(k, v)
        incremental_cache = self.self_attn.gen_cache(k, v)
        return incremental_cache

class ErnieVilgEncoder(nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(ErnieVilgEncoder, self).__init__(encoder_layer, num_layers, norm)
        self.layers = nn.LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    
    def forward(self, src, src_masks=None, caches=None):
        
        # src_mask = _convert_attention_mask(src_mask, src.dtype)
        output = src
        new_caches = []
        num_layers = len(self.layers)
        
        for i, mod in enumerate(self.layers):
            if i == num_layers - 1:
                attn_id = 2
            elif (i - 1) % 4 == 0:
                attn_id = 1
            else:
                attn_id = 0

            cache = None
            if caches:
                cache = caches[i]
                # q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
                # q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
                # print('output shape:', output.shape)
                cache['k'] = paddle.reshape(cache['k'], [0, 0, mod.self_attn.num_heads, mod.self_attn.head_dim])
                cache['k'] = paddle.transpose(cache['k'], [0, 2, 1, 3])
                cache['v'] = paddle.reshape(cache['v'], [0, 0, mod.self_attn.num_heads, mod.self_attn.head_dim])
                cache['v'] = paddle.transpose(cache['v'], [0, 2, 1, 3])
                cache = mod.gen_cache(cache['k'], cache['v'])
                # print('layer', i, 'shape:', cache.k.shape, cache.v.shape)
                output, new_cache = mod(output,
                                        src_mask=src_masks[attn_id],
                                        cache=cache)
                new_caches.append(new_cache)
            # break
        print(output[0,0])    
        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    

@register_base_model
class ErnieVilGModel(ErnieVilGPretrainedModel):
    """
    The bare UnifiedTransformer Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a `paddle.nn.Layer <https://www.paddlepaddle.org.cn
    /documentation/docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ 
    subclass. Use it as a regular Paddle Layer and refer to the Paddle 
    documentation for all matter related to general usage and behavior.
    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in :class:`UnifiedTransformerModel`. 
            Also is the vocab size of token embedding matrix.
        hidden_size (int, optional):
            Dimensionality of the embedding layers, encoder layers and pooler 
            layer. Defaults to 768.
        num_hidden_layers (int, optional):
            The number of hidden layers in the encoder. Defaults to 12.
        num_attention_heads (int, optional):
            The number of heads in multi-head attention(MHA). Defaults to 12.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward layer in the encoder. Input 
            tensors to feed-forward layers are firstly projected from 
            `hidden_size` to `intermediate_size`, and then projected back to 
            `hidden_size`. Typically `intermediate_size` is larger than 
            `hidden_size`. Defaults to 3072.
        hidden_act (str, optional):
            The activation function in the feedforward network. Defaults to 
            "gelu".
        hidden_dropout_prob(float, optional): 
            The dropout probability used in pre-process and post-precess of MHA 
            and FFN sub-layer. Defaults to 0.1.
        attention_probs_dropout_prob (float, optional): 
            The dropout probability used in MHA to drop some attention target. 
            Defaults to 0.1.
        normalize_before (bool, optional): 
            Indicate whether to put layer normalization into preprocessing of 
            MHA and FFN sub-layers. If True, pre-process is layer normalization 
            and post-precess includes dropout, residual connection. Otherwise, 
            no pre-process and post-precess includes dropout, residual 
            connection, layer normalization. Defaults to True.
        max_position_embeddings (int, optional):
            The maximum length of input `position_ids`. Defaults to 512.
        type_vocab_size (int, optional):
            The size of the input `token_type_ids`. Defaults to 2.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.
            .. note::
                A normal_initializer initializes weight matrices as normal 
                distributions. See 
                :meth:`UnifiedTransformerPretrainedModel.init_weights` method 
                for how weights are initialized in 
                :class:`UnifiedTransformerModel`.
        unk_token_id (int, optional):
            The id of special token `unk_token`. Defaults to 0.
        pad_token_id (int, optional):
            The id of special token `pad_token`. Defaults to 0.
        bos_token_id (int, optional):
            The id of special token `bos_token`. Defaults to 1.
        eos_token_id (int, optional):
            The id of special token `eos_token`. Defaults to 2.
        mask_token_id (int, optional):
            The id of special token `mask_token`. Defaults to 30000.
    """

    def __init__(
            self,
            vocab_size=19000,
            img_vocab_size=8194,
            hidden_size=4096,
            # intermediate_hidden_size=16384,
            max_seq_len=34,
            rel_pos_bin=32,
            num_hidden_layers=48,
            num_attention_heads=64,
            intermediate_size=16384,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            normalize_before=True,
            max_position_embeddings=513,
            type_vocab_size=2,
            initializer_range=0.02,
            unk_token_id=0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=30000, ):
        super(ErnieVilGModel, self).__init__()
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        intermediate_hidden_size = hidden_size * 4
        emb_size = hidden_size

        self.embeddings = ErnieVilGEmbeddings(
            vocab_size, img_vocab_size, 
                 hidden_size=emb_size,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=max_position_embeddings,
                 max_img_row=34,
                 max_img_col=34,)
        # encoder_layer = nn.TransformerEncoderLayer(
        encoder_layer = ErnieVilgEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0,
            normalize_before=normalize_before)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder = ErnieVilgEncoder(encoder_layer, num_hidden_layers,
                                             encoder_norm)
        self.lm_trans_fc = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.lm_trans_layer_norm = nn.LayerNorm(hidden_size)
        self.lm_out_fc = nn.Linear(hidden_size, img_vocab_size)

        self.apply(self.init_weights)

    def decode(self, text_ids, pos_ids, img_ids, img_row_pos_ids, img_col_pos_ids, is_encoding, input_masks, cache_idxes, cache_text, cache_image):
        embedding_output = self.embeddings(text_ids, img_ids, pos_ids, img_row_pos_ids, img_col_pos_ids, is_encoding)
        
        merged_caches = []
        for l, text_kv, image_kv in zip(range(self.num_hidden_layers), cache_text, cache_image):
            if l == self.num_hidden_layers - 1:
                attn_id = 2
            elif (l - 1) % 4 == 0:
                attn_id = 1
            else:
                attn_id = 0

            cache_idx = cache_idxes[attn_id]
            image_k = paddle.gather_nd(image_kv[0], cache_idx)
            image_v = paddle.gather_nd(image_kv[1], cache_idx)
            merged_caches.append({"k": paddle.concat([text_kv[0], image_k], axis=1),
                                  "v": paddle.concat([text_kv[1], image_v], axis=1)})

        self_attn_mask = map(lambda x: paddle.fluid.layers.scale(
            x=x, scale=10000.0, bias=-1.0, bias_after_scale=False), input_masks)
        n_head_self_attn_mask = map(lambda x: paddle.fluid.layers.stack(
            x=[x] * self.num_attention_heads, axis=1), self_attn_mask)

        n_head_self_attn_mask = list(n_head_self_attn_mask)                             
        
        sequence_output, cache = self.encoder(embedding_output,
                                              n_head_self_attn_mask, merged_caches)
        sequence_output = sequence_output.reshape([-1, self.hidden_size])
        logit = self.lm_trans_fc(sequence_output)
        logit = self.act(logit)
        logit = self.lm_trans_layer_norm(logit)
        logit = self.lm_out_fc(logit)
        probs = paddle.fluid.layers.softmax(logit[:,:-2] / 1.0)
        sampling_ids = paddle.fluid.layers.sampling_id(probs, dtype="int")
        sampling_scores = paddle.fluid.layers.one_hot(
            paddle.fluid.layers.unsqueeze(sampling_ids, [1]), probs.shape[1]
        )
        sampling_scores = sampling_scores * probs - (1 - sampling_scores) * 1e3
        topk_scores, topk_indices = paddle.fluid.layers.topk(
            input=sampling_scores, k=1)
        return topk_scores, topk_indices, logit, cache


    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                use_cache=False,
                cache=None):
        r"""
        The UnifiedTransformerModel forward method, overrides the special 
        :meth:`__call__` method.
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input 
                sequence. It's data type should be `int64` and has a shape of 
                [batch_size, sequence_length].
            token_type_ids (Tensor):
                Segment token indices to indicate first and second portions of 
                the inputs. Indices can be either 0 or 1:
                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.
                It's data type should be `int64` and has a shape of 
                [batch_size, sequence_length].
            position_ids (Tensor):
                The position indices of input sequence tokens. It's data type 
                should be `int64` and has a shape of [batch_size, sequence_length].
            attention_mask (Tensor): 
                A tensor used in multi-head attention to prevents attention to 
                some unwanted positions, usually the paddings or the subsequent 
                positions. It is a tensor with shape broadcasted to 
                [batch_size, n_head, sequence_length, sequence_length]. 
                
                - When the data type is bool, the unwanted positions have 
                  `False` values and the others have `True` values. 
                - When the data type is int, the unwanted positions have 0 
                  values and the others have 1 values. 
                - When the data type is float, the unwanted positions have 
                  `-INF` values and the others have 0 values.
            use_cache: (bool, optional): 
                Whether or not use the model cache to speed up decoding. Defaults 
                to False.
            cache (list, optional): 
                It is a list, and each element in the list is `incremental_cache` 
                produced by :meth:`paddle.nn.TransformerEncoderLayer.gen_cache` 
                method. See :meth:`paddle.nn.TransformerEncoder.gen_cache` 
                method for more details. It is only used for inference and 
                should be None for training. Defaults to None.
        Returns:
            Tensor|tuple: If `use_cache` is False, it is a tensor 
            representing the output of :class:`UnifiedTransformerModel`, with 
            shape [batch_size, sequence_length, hidden_size]. The data type is 
            float32 or float64. Otherwise, it is a tuple, besides the output of 
            :class:`UnifiedTransformerModel`, the tuple also includes the new 
            cache which is same as input `cache` but `incremental_cache` in it 
            has an incremental length. 
            See :meth:`paddle.nn.MultiHeadAttention.gen_cache` method and 
            :meth:`paddle.nn.MultiHeadAttention.forward` method for more details.
        Example:
            .. code-block::
                from paddlenlp.transformers import UnifiedTransformerModel
                from paddlenlp.transformers import UnifiedTransformerTokenizer
                model = UnifiedTransformerModel.from_pretrained('plato-mini')
                tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
                history = '我爱祖国'
                inputs = tokenizer.dialogue_encode(
                    history,
                    return_tensors=True,
                    is_split_into_words=False)
                outputs = model(**inputs)
        """

        embedding_output = self.embeddings(input_ids, token_type_ids,
                                           position_ids)
        if use_cache:
            if cache is None:
                cache = self.encoder.gen_cache(embedding_output)
            sequence_output, cache = self.encoder(embedding_output,
                                                  attention_mask, cache)

            return sequence_output, cache
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)

            return sequence_output


# class UnifiedTransformerLMHead(nn.Layer):
#     def __init__(self,
#                  hidden_size,
#                  vocab_size,
#                  activation,
#                  embedding_weights=None):
#         super(UnifiedTransformerLMHead, self).__init__()
#         self.transform = nn.Linear(hidden_size, hidden_size)
#         self.activation = getattr(nn.functional, activation)
#         self.layer_norm = nn.LayerNorm(hidden_size)
#         self.decoder_weight = self.create_parameter(
#             shape=[vocab_size, hidden_size],
#             dtype=self.transform.weight.dtype,
#             is_bias=False) if embedding_weights is None else embedding_weights
#         self.decoder_bias = self.create_parameter(
#             shape=[vocab_size], dtype=self.decoder_weight.dtype, is_bias=True)

#     def forward(self, hidden_states, masked_positions=None):
#         if masked_positions is not None:
#             hidden_states = paddle.reshape(hidden_states,
#                                            [-1, hidden_states.shape[-1]])
#             hidden_states = paddle.tensor.gather(hidden_states,
#                                                  masked_positions)
#         hidden_states = self.transform(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.layer_norm(hidden_states)
#         logits = paddle.tensor.matmul(
#             hidden_states, self.decoder_weight,
#             transpose_y=True) + self.decoder_bias
#         return logits


# class UnifiedTransformerLMHeadModel(UnifiedTransformerPretrainedModel):
#     """
#     The UnifiedTransformer Model with a language modeling head on top
#     for generation tasks.
#     Args:
#         unified_transformer (:class:`UnifiedTransformerModel`):
#             An instance of :class:`UnifiedTransformerModel`.
#     """

#     def __init__(self, unified_transformer):
#         super(UnifiedTransformerLMHeadModel, self).__init__()
#         self.unified_transformer = unified_transformer
#         self.lm_head = UnifiedTransformerLMHead(
#             self.unified_transformer.config["hidden_size"],
#             self.unified_transformer.config["vocab_size"],
#             self.unified_transformer.config["hidden_act"],
#             self.unified_transformer.embeddings.word_embeddings.weight)
#         self.apply(self.init_weights)

#     def forward(self,
#                 input_ids,
#                 token_type_ids,
#                 position_ids,
#                 attention_mask,
#                 masked_positions=None,
#                 use_cache=False,
#                 cache=None):
#         r"""
#         The UnifiedTransformerLMHeadModel forward method, overrides the special 
#         :meth:`__call__` method.
#         Args:
#             input_ids (Tensor):
#                 See :class:`UnifiedTransformerModel`.
#             token_type_ids (Tensor):
#                 See :class:`UnifiedTransformerModel`.
#             position_ids (Tensor):
#                 See :class:`UnifiedTransformerModel`.
#             attention_mask (Tensor): 
#                 See :class:`UnifiedTransformerModel`.
#             use_cache: (bool, optional): 
#                 See :class:`UnifiedTransformerModel`.
#             cache (list, optional): 
#                 See :class:`UnifiedTransformerModel`.
#         Returns:
#             Tensor|tuple: If `use_cache` is False, it is a tensor 
#             representing the output of :class:`UnifiedTransformerLMHeadModel`, 
#             with shape [batch_size, sequence_length, vocab_size]. The data type 
#             is float32 or float64. Otherwise, it is a tuple, besides the output 
#             of :class:`UnifiedTransformerLMHeadModel`, the tuple also includes 
#             the new cache which is same as input `cache` but `incremental_cache` 
#             in it has an incremental length. 
#             See :meth:`paddle.nn.MultiHeadAttention.gen_cache` method and 
#             :meth:`paddle.nn.MultiHeadAttention.forward` method for more details.
#         Example:
#             .. code-block::
#                 from paddlenlp.transformers import UnifiedTransformerLMHeadModel
#                 from paddlenlp.transformers import UnifiedTransformerTokenizer
#                 model = UnifiedTransformerLMHeadModel.from_pretrained('plato-mini')
#                 tokenizer = UnifiedTransformerTokenizer.from_pretrained('plato-mini')
#                 history = '我爱祖国'
#                 inputs = tokenizer.dialogue_encode(
#                     history,
#                     return_tensors=True,
#                     is_split_into_words=False)
#                 logits = model(**inputs)
#         """

#         outputs = self.unified_transformer(input_ids, token_type_ids,
#                                            position_ids, attention_mask,
#                                            use_cache, cache)
#         sequence_output = outputs[0] if use_cache else outputs
#         logits = self.lm_head(sequence_output, masked_positions)
#         if use_cache:
#             cache = outputs[1]
#             return logits, cache
#         else:
#             return logits

#     def prepare_faster_entry(self, kwargs):
#         from paddlenlp.ops import FasterUnifiedTransformer
#         use_fp16_decoding = kwargs.get('use_fp16_decoding', False)
#         decode_strategy = kwargs.get('decode_strategy')
#         if decode_strategy == 'sampling' and kwargs.get(
#                 'top_k') != 0 and kwargs.get('top_p') != 1:
#             raise AttributeError(
#                     "Only topk sampling or topp sampling are supported. " \
#                     "Topk sampling and topp sampling cannot be both applied in the faster version.")
#         if kwargs['repetition_penalty'] != 1.0:
#             # not support for repetition_penalty yet in the faster version
#             raise AttributeError(
#                 "'repetition_penalty != 1' is not supported yet in the faster version"
#             )
#         if kwargs['forced_bos_token_id'] is not None:
#             # not support for min_length yet in the faster version
#             raise AttributeError(
#                 "'forced_bos_token_id != None' is not supported yet in the faster version"
#             )
#         self._faster_entry = FasterUnifiedTransformer(
#             self, use_fp16_decoding=use_fp16_decoding).forward
#         return self._faster_entry

#     def adjust_logits_during_generation(self, logits):
#         # pre-process distribution
#         logits[:, self.unified_transformer.unk_token_id] = -1e9
#         logits[:, self.unified_transformer.bos_token_id] = -1e9
#         logits[:, self.unified_transformer.mask_token_id] = -1e9
#         return logits

#     def prepare_inputs_for_generation(self,
#                                       input_ids,
#                                       token_type_ids,
#                                       position_ids,
#                                       attention_mask,
#                                       use_cache=False,
#                                       cache=None,
#                                       **kwargs):
#         # only last token for inputs_ids if cache is defined in kwargs
#         if cache is not None:
#             input_ids = input_ids[:, -1].unsqueeze(-1)
#             token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
#             position_ids = position_ids[:, -1].unsqueeze(-1)
#             attention_mask = attention_mask[:, :, -1, :].unsqueeze(2)

#         return {
#             "input_ids": input_ids,
#             "token_type_ids": token_type_ids,
#             "position_ids": position_ids,
#             "attention_mask": attention_mask,
#             "use_cache": use_cache,
#             "cache": cache
#         }

#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError as e:
#             try:
#                 return getattr(getattr(self, self.base_model_prefix), name)
#             except AttributeError:
#                 try:
#                     return getattr(self, self.base_model_prefix).config[name]
#                 except KeyError:
#                     raise e


# UnifiedTransformerForMaskedLM = UnifiedTransformerLMHeadModel

# if __name__=='__main__':
#     model = ErnieVilGModel()
#     paddle.save(model.state_dict(), 'ernie_vilg_dynamic.pdparams')