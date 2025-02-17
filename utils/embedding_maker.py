__all__ = [
    'create_embeddings', 'load_embedding', 'load_vocab',
    'encoding_and_padding', 'get_embedding_model'
]

import bz2
import json
import os

import numpy as np
import pkg_resources
from gensim.models import Word2Vec

from utils.spacing_utils import sent_to_spacing_chars

"""
@param sequences 공백없는 훈련용 문장 배열
@param maxlen 200
@
"""
def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))  # for recognizing the longest sentence length

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)    # the longest sentence length

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' %
                             truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def get_sentence_generator(data_dir,
                            splitc=' '):
    class SentenceGenerator(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):
            print('print __iter__')
            for fname in os.listdir(self.dirname):
                print("processing~  '{}'".format(fname))
                for line in bz2.open(os.path.join(self.dirname, fname), "rt"):
                    yield sent_to_spacing_chars(line.strip()).split(splitc)
        
        def size(self):
            print('print SentenceGenerator.size()')
            num_lines = 0
            for fname in os.listdir(self.dirname):
                num_lines = sum(1 for _ in bz2.open(os.path.join(self.dirname, fname)))

            return num_lines

    return SentenceGenerator(data_dir)

def update_full_model(data_dir,
                      model_file,
                      embeddings_file,
                      vocab_file,
                      **params):
    
            
    
    model = Word2Vec.load(model_file)

    sentences = get_sentence_generator(data_dir)

    print("model.corpus_count==" + str(model.corpus_count))
    print("model.epochs==" + str(model.epochs))

    # 아래 명령을 넣으면 total_examples 가 확 줄어든다;;;; 파일용량에는 변함이 없음
    model.build_vocab(sentences, update=True)

    model.train(sentences,
                total_examples=(model.corpus_count + sentences.size()),
                epochs=model.epochs)

    # model.train(model.train(sentences, epochs=model.epochs, 
    #         total_examples=model.corpus_count,
    #         start_alpha=(model.min_alpha + (53/200 * (model.alpha - model.min_alpha)))


    model.save(model_file)

    overwrite_embedding_w2v_file(model, embeddings_file, vocab_file)


def create_embeddings(data_dir,
                      model_file,
                      embeddings_file,
                      vocab_file,
                      **params):
    """
    making embedding from files.
    :**params additional Word2Vec() parameters
    :splitc   char for splitting in  data_dir files
    :model_file output object from Word2Vec()
    :data_dir data dir to be process
    :embeddings_file numpy object file path from Word2Vec()
    :vocab_file item to index json dictionary
    """

    sentences = get_sentence_generator(data_dir)
    
    model = Word2Vec(sentences, **params)
    model.save(model_file)
    
    overwrite_embedding_w2v_file(model, embeddings_file, vocab_file)


def overwrite_embedding_w2v_file(model, embeddings_file, vocab_file):
    weights = model.wv.vectors  # previously known as model.wv.syn0
    default_vec = np.mean(weights, axis=0, keepdims=True)
    padding_vec = np.zeros((1, weights.shape[1]))

    weights_default = np.concatenate([weights, default_vec, padding_vec],
                                     axis=0)

    np.save(open(embeddings_file, 'wb'), weights_default)

    #vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    vocab = dict([(k, v) for k, v in model.wv.key_to_index.items()])    # model.wv.vocab is deprecated
    vocab['__ETC__'] = weights_default.shape[0] - 2
    vocab['__PAD__'] = weights_default.shape[0] - 1
    with open(vocab_file, 'w') as f:
        f.write(json.dumps(vocab))


def load_embedding(embeddings_file):
    return (np.load(embeddings_file))


# returns (word: index) sets and (index: word) sets
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

"""
@param word2idx_dic
@param sequences 공백없는 훈련용 문장 배열
"""
def encoding_and_padding(word2idx_dic, sequences, **params):
    """
    1. making item to idx
    2. padding
    :word2idx_dic
    :sequences: list of lists where each element is a sequence
    :maxlen: int, maximum length
    :dtype: type to cast the resulting sequence.
    :padding: 'pre' or 'post', pad either before or after each sequence.
    :truncating: 'pre' or 'post', remove values from sequences larger than
        maxlen either in the beginning or in the end of the sequence
    :value: float, value to pad the sequences to the desired value.
    """
    seq_idx = [[word2idx_dic.get(a, word2idx_dic['__ETC__']) for a in i]
               for i in sequences]
    params['value'] = word2idx_dic['__PAD__']
    return (pad_sequences(seq_idx, **params))


def get_embedding_model(name='fee_prods', path='data/embedding'):
    weights = pkg_resources.resource_filename(
        'dsc', os.path.join(path, name, 'weights.np'))
    w2idx = pkg_resources.resource_filename(
        'dsc', os.path.join(path, name, 'idx.json'))
    return ((load_embedding(weights), load_vocab(w2idx)[0]))
