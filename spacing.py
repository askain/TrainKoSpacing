import argparse
import bz2
import logging
import re
import time

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import gluonnlp as nlp
import mxnet as mx
import mxnet.autograd as autograd
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn, rnn
from tqdm import tqdm

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


def y_encoding(n_grams, maxlen=200):
    # 입력된 문장으로 정답셋 인코딩함
    init_mat = np.zeros(shape=(len(n_grams), maxlen), dtype=np.int8)
    for i in range(len(n_grams)):
        init_mat[i, np.cumsum([len(j) for j in n_grams[i]]) - 1] = 1
    return init_mat


def split_train_set(x_train, p=0.98):
    """
    > split_train_set(pd.DataFrame({'a':[1,2,3,4,None], 'b':[5,6,7,8,9]}))
    (array([0, 4, 3]), [1, 2])
    """
    import numpy as np
    train_idx = np.random.choice(range(x_train.shape[0]),
                                 int(x_train.shape[0] * p),
                                 replace=False)
    set_tr_idx = set(train_idx)
    test_index = [i for i in range(x_train.shape[0]) if i not in set_tr_idx]
    return ((train_idx, np.array(test_index)))


def get_generator(x, y, batch_size):
    tr_set = gluon.data.ArrayDataset(x, y.astype('float32'))
    tr_data_iterator = gluon.data.DataLoader(tr_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_workers)
    return (tr_data_iterator)


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


def model_init(n_hidden, vocab_size, embed_dim, max_seq_length, ctx):
    # 모형 인스턴스 생성 및 트래이너, loss 정의
    # n_hidden, vocab_size, embed_dim, max_seq_length
    model = pick_model(opt.model_type, n_hidden, vocab_size, embed_dim, max_seq_length)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    model.embedding.weight.set_data(weights)
    model.hybridize(static_alloc=True)
    # 임베딩 영역 가중치 고정
    model.embedding.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(model.collect_params(), 'rmsprop')
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    loss.hybridize(static_alloc=True)
    return (model, loss, trainer)


def evaluate_accuracy(data_iterator, net, pad_idx, ctx, n=5000):
    # 각 시퀀스의 길이만큼 순회하며 정확도 측정
    # 최적화되지 않음
    acc = mx.metric.Accuracy(axis=0)
    num_of_test = 0
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # get sentence length
        data_np = data.asnumpy()
        lengths = np.argmax(np.where(data_np == pad_idx, np.ones_like(data_np),
                                     np.zeros_like(data_np)),
                            axis=1)
        output = net(data)
        pred_label = output.squeeze(axis=2) > 0.5

        for i in range(data.shape[0]):
            num_of_test += data.shape[0]
            acc.update(preds=pred_label[i, :lengths[i]],
                       labels=label[i, :lengths[i]])
        if num_of_test > n:
            break
    return acc.get()[1]


def train(epochs,
          tr_data_iterator,
          te_data_iterator,
          va_data_iterator,
          model,
          loss,
          trainer,
          pad_idx,
          ctx,
          mdl_desc="spacing_model",
          decay=False):
    # 학습 코드
    tot_test_acc = []
    tot_train_loss = []
    for e in range(epochs):
        tic = time.time()
        # Decay learning rate.
        if e > 1 and decay:
            trainer.set_learning_rate(trainer.learning_rate * 0.7)
        train_loss = []
        iter_tqdm = tqdm(tr_data_iterator, 'Batches')
        for i, (x_data, y_data) in enumerate(iter_tqdm):
            x_data_l = gluon.utils.split_and_load(x_data,
                                                  ctx,
                                                  even_split=False)
            y_data_l = gluon.utils.split_and_load(y_data,
                                                  ctx,
                                                  even_split=False)

            with autograd.record():
                losses = [
                    loss(model(x), y) for x, y in zip(x_data_l, y_data_l)
                ]
            for l in losses:
                l.backward()
            trainer.step(x_data.shape[0])
            curr_loss = np.mean([mx.nd.mean(l).asscalar() for l in losses])
            train_loss.append(curr_loss)
            iter_tqdm.set_description("loss {}".format(curr_loss))
            mx.nd.waitall()

        # caculate test loss
        test_acc = evaluate_accuracy(
            te_data_iterator,
            model,
            pad_idx,
            ctx=ctx[0] if isinstance(ctx, list) else mx.gpu(0))
        valid_acc = evaluate_accuracy(
            va_data_iterator,
            model,
            pad_idx,
            ctx=ctx[0] if isinstance(ctx, list) else mx.gpu(0))
        logger.info('[Epoch %d] time cost: %f' % (e, time.time() - tic))
        logger.info("[Epoch %d] Train Loss: %f, Test acc : %f Valid acc : %f" %
                    (e, np.mean(train_loss), test_acc, valid_acc))
        tot_test_acc.append(test_acc)
        tot_train_loss.append(np.mean(train_loss))
        model.save_parameters(opt.outputs + '/' + "{}_{}.params".format(mdl_desc, e))
    return (tot_test_acc, tot_train_loss)


def pre_processing(setences):
    # 공백은 ^
    char_list = [li.strip().replace(' ', '^') for li in setences]
    # 문장의 시작 포인트 «
    # 문장의 끌 포인트  »
    char_list = ["«" + li + "»" for li in char_list]
    # 문장 -> 문자열
    char_list = [''.join(list(li)) for li in char_list]
    return char_list


def make_input_data(inputs,
                    train_ratio,
                    sampling,
                    make_lag_set=False,
                    batch_size=200):
    with bz2.open(inputs, 'rt') as f:
        line_list = [i.strip() for i in f.readlines() if i.strip() != '']
    logger.info('complete loading train file!')

    # 아버지가 방에 들어가신다. -> '«아버지가^방에^들어가신다.»'
    processed_seq = pre_processing(line_list)
    logger.info(processed_seq[0])
    # n percent random sample
    logger.info('random sampling on training set!')
    samp_idx = np.random.choice(range(len(processed_seq)),
                                int(len(processed_seq) * sampling),
                                replace=False)
    processed_seq_samp = [processed_seq[i] for i in samp_idx]
    sp_sents = [i.split('^') for i in processed_seq_samp]

    sp_sents = list(filter(lambda x: len(x) >= 8, sp_sents))

    # max 8 어절 씩 1어절 shift하여 학습 데이터 생성
    if make_lag_set is True:
        n_gram = [[k, v, z, a, c, d, e, f]
                  for sent in sp_sents for k, v, z, a, c, d, e, f in zip(
                      sent, sent[1:], sent[2:], sent[3:], sent[4:], sent[5:],
                      sent[6:], sent[7:])]
    else:
        n_gram = sp_sents
    # max 200문자 이하만 사용
    n_gram = [i for i in n_gram if len("^".join(i)) <= opt.max_seq_len]
    # y 정답 인코딩
    n_gram_y = y_encoding(n_gram, opt.max_seq_len)
    logger.info(n_gram[0])
    logger.info(n_gram_y[0])
    # vocab file 로딩
    w2idx, _ = load_vocab(opt.vocab_file)

    # 학습셋을 만들기 위해 공백을 제거하고 문자 인덱스로 인코딩함
    logger.info('index eocoding!')
    ngram_coding_seq = encoding_and_padding(
        word2idx_dic=w2idx,
        sequences=[''.join(gram) for gram in n_gram],
        maxlen=opt.max_seq_len,
        padding='post',
        truncating='post')
    logger.info(ngram_coding_seq[0])
    if train_ratio < 1:
        # 학습셋 테스트셋 생성
        tr_idx, te_idx = split_train_set(ngram_coding_seq, train_ratio)

        y_train = n_gram_y[tr_idx, ]
        x_train = ngram_coding_seq[tr_idx, ]

        y_test = n_gram_y[te_idx, ]
        x_test = ngram_coding_seq[te_idx, ]

        # train generator
        train_generator = get_generator(x_train, y_train, batch_size)
        valid_generator = get_generator(x_test, y_test, 500)
        return (train_generator, valid_generator)
    else:
        train_generator = get_generator(ngram_coding_seq, n_gram_y, batch_size)
        return (train_generator)


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
