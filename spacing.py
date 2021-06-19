import argparse
import re

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn, rnn

from utils.embedding_maker import (encoding_and_padding, load_embedding,
                                   load_vocab)

GPU_COUNT = 1
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

parser = argparse.ArgumentParser(description='Korean Autospacing Trainer')

def spacing(sent):
    # 사전 파일 로딩
    w2idx, idx2w = load_vocab('model/w2idx.dic')
    # 임베딩 파일 로딩
    weights = load_embedding('model/kospacing_wv.np')
    vocab_size = weights.shape[0]
    embed_dim = weights.shape[1]
    model = pick_model('kospacing', 200, vocab_size, embed_dim, 200)

    # model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu(0))
    # model.embedding.weight.set_data(weights)
    model.load_parameters('model/kospacing.params', ctx=mx.cpu(0))
    predictor = pred_spacing(model, w2idx)

    return predictor.get_spaced_sent(sent, 200)

# Model class
class korean_autospacing_base(gluon.HybridBlock):
    def __init__(self, n_hidden, vocab_size, embed_dim, max_seq_length,
                 **kwargs):
        super(korean_autospacing_base, self).__init__(**kwargs)
        # 입력 시퀀스 길이
        self.in_seq_len = max_seq_length
        # 출력 시퀀스 길이
        self.out_seq_len = max_seq_length
        # GRU의 hidden 개수
        self.n_hidden = n_hidden
        # 고유문자개수
        self.vocab_size = vocab_size
        # max_seq_length
        self.max_seq_length = max_seq_length
        # 임베딩 차원수
        self.embed_dim = embed_dim

        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=self.vocab_size,
                                          output_dim=self.embed_dim)

            self.conv_unigram = nn.Conv2D(channels=128,
                                          kernel_size=(1, self.embed_dim))

            self.conv_bigram = nn.Conv2D(channels=256,
                                         kernel_size=(2, self.embed_dim),
                                         padding=(1, 0))

            self.conv_trigram = nn.Conv2D(channels=128,
                                          kernel_size=(3, self.embed_dim),
                                          padding=(1, 0))

            self.conv_forthgram = nn.Conv2D(channels=64,
                                            kernel_size=(4, self.embed_dim),
                                            padding=(2, 0))

            self.conv_fifthgram = nn.Conv2D(channels=32,
                                            kernel_size=(5, self.embed_dim),
                                            padding=(2, 0))

            self.bi_gru = rnn.GRU(hidden_size=self.n_hidden, layout='NTC', bidirectional=True)
            self.dense_sh = nn.Dense(100, activation='relu', flatten=False)
            self.dense = nn.Dense(1, activation='sigmoid', flatten=False)

    def hybrid_forward(self, F, inputs):
        embed = self.embedding(inputs)
        embed = F.expand_dims(embed, axis=1)
        unigram = self.conv_unigram(embed)
        bigram = self.conv_bigram(embed)
        trigram = self.conv_trigram(embed)
        forthgram = self.conv_forthgram(embed)
        fifthgram = self.conv_fifthgram(embed)

        grams = F.concat(unigram,
                         F.slice_axis(bigram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         trigram,
                         F.slice_axis(forthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(fifthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         dim=1)

        grams = F.transpose(grams, (0, 2, 3, 1))
        grams = F.reshape(grams, (-1, self.max_seq_length, -3))
        grams = self.bi_gru(grams)
        fc1 = self.dense_sh(grams)
        return (self.dense(fc1))


# https://raw.githubusercontent.com/haven-jeon/Train_KoSpacing/master/img/kosapcing_img.png
class korean_autospacing2(gluon.HybridBlock):
    def __init__(self, n_hidden, vocab_size, embed_dim, max_seq_length,
                 **kwargs):
        super(korean_autospacing2, self).__init__(**kwargs)
        # 입력 시퀀스 길이
        self.in_seq_len = max_seq_length
        # 출력 시퀀스 길이
        self.out_seq_len = max_seq_length
        # GRU의 hidden 개수
        self.n_hidden = n_hidden
        # 고유문자개수
        self.vocab_size = vocab_size
        # max_seq_length
        self.max_seq_length = max_seq_length
        # 임베딩 차원수
        self.embed_dim = embed_dim

        with self.name_scope():
            self.embedding = nn.Embedding(input_dim=self.vocab_size,
                                          output_dim=self.embed_dim)

            self.conv_unigram = nn.Conv2D(channels=128,
                                          kernel_size=(1, self.embed_dim))

            self.conv_bigram = nn.Conv2D(channels=128,
                                         kernel_size=(2, self.embed_dim),
                                         padding=(1, 0))

            self.conv_trigram = nn.Conv2D(channels=64,
                                          kernel_size=(3, self.embed_dim),
                                          padding=(2, 0))

            self.conv_forthgram = nn.Conv2D(channels=32,
                                            kernel_size=(4, self.embed_dim),
                                            padding=(3, 0))

            self.conv_fifthgram = nn.Conv2D(channels=16,
                                            kernel_size=(5, self.embed_dim),
                                            padding=(4, 0))
            # for reverse convolution
            self.conv_rev_bigram = nn.Conv2D(channels=128,
                                             kernel_size=(2, self.embed_dim),
                                             padding=(1, 0))

            self.conv_rev_trigram = nn.Conv2D(channels=64,
                                              kernel_size=(3, self.embed_dim),
                                              padding=(2, 0))

            self.conv_rev_forthgram = nn.Conv2D(channels=32,
                                                kernel_size=(4,
                                                             self.embed_dim),
                                                padding=(3, 0))

            self.conv_rev_fifthgram = nn.Conv2D(channels=16,
                                                kernel_size=(5,
                                                             self.embed_dim),
                                                padding=(4, 0))
            self.bi_gru = rnn.GRU(hidden_size=self.n_hidden, layout='NTC', bidirectional=True)
            # self.bi_gru = rnn.BidirectionalCell(
            #     rnn.GRUCell(hidden_size=self.n_hidden),
            #     rnn.GRUCell(hidden_size=self.n_hidden))
            self.dense_sh = nn.Dense(100, activation='relu', flatten=False)
            self.dense = nn.Dense(1, activation='sigmoid', flatten=False)

    def hybrid_forward(self, F, inputs):
        embed = self.embedding(inputs)
        embed = F.expand_dims(embed, axis=1)
        rev_embed = embed.flip(axis=2)

        unigram = self.conv_unigram(embed)
        bigram = self.conv_bigram(embed)
        trigram = self.conv_trigram(embed)
        forthgram = self.conv_forthgram(embed)
        fifthgram = self.conv_fifthgram(embed)

        rev_bigram = self.conv_rev_bigram(rev_embed).flip(axis=2)
        rev_trigram = self.conv_rev_trigram(rev_embed).flip(axis=2)
        rev_forthgram = self.conv_rev_forthgram(rev_embed).flip(axis=2)
        rev_fifthgram = self.conv_rev_fifthgram(rev_embed).flip(axis=2)

        grams = F.concat(unigram,
                         F.slice_axis(bigram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(rev_bigram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(trigram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(rev_trigram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(forthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(rev_forthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(fifthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         F.slice_axis(rev_fifthgram,
                                      axis=2,
                                      begin=0,
                                      end=self.max_seq_length),
                         dim=1)

        grams = F.transpose(grams, (0, 2, 3, 1))
        grams = F.reshape(grams, (-1, self.max_seq_length, -3))
        grams = self.bi_gru(grams)
        fc1 = self.dense_sh(grams)
        return (self.dense(fc1))


def pick_model(model_nm, n_hidden, vocab_size, embed_dim, max_seq_length):
    if model_nm.lower() == 'kospacing':
        model = korean_autospacing_base(n_hidden=n_hidden,
                                        vocab_size=vocab_size,
                                        embed_dim=embed_dim,
                                        max_seq_length=max_seq_length)
    elif model_nm.lower() == 'kospacing2':
        model = korean_autospacing2(n_hidden=n_hidden,
                                    vocab_size=vocab_size,
                                    embed_dim=embed_dim,
                                    max_seq_length=max_seq_length)
    else:
        assert False
    return model


class pred_spacing:
    def __init__(self, model, w2idx):
        self.model = model
        self.w2idx = w2idx
        self.pattern = re.compile(r'\s+')

    @lru_cache(maxsize=None)
    def get_spaced_sent(self, raw_sent, max_seq_len):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [
            raw_sent_,
        ]
        mat_in = encoding_and_padding(word2idx_dic=self.w2idx,
                                      sequences=sents_in,
                                      maxlen=max_seq_len,
                                      padding='post',
                                      truncating='post')
        mat_in = mx.nd.array(mat_in, ctx=mx.cpu(0))
        results = self.model(mat_in)
        mat_set = results[0, ]
        preds = np.array(
            ['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        return self.make_pred_sents(raw_sent_, preds)

    def make_pred_sents(self, x_sents, y_pred):
        res_sent = []
        for i, j in zip(x_sents, y_pred):
            if j == '1':
                res_sent.append(i)
                res_sent.append(' ')
            else:
                res_sent.append(i)
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs
