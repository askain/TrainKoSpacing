# coding=utf-8
# Copyright 2020 Heewon Jeon. All rights reserved.
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
from utils.embedding_maker import create_embeddings, update_full_model


parser = argparse.ArgumentParser(description='Korean Autospacing Embedding Maker')

# parser.add_argument('--num-iters', type=int, default=5,
#                     help='number of iterations to train (default: 5)')

parser.add_argument('--min-count', type=int, default=100,
                    help='mininum word counts to filter (default: 100)')

# parser.add_argument('--embedding-size', type=int, default=100,
#                     help='embedding dimention size (default: 100)')

parser.add_argument('--num-worker', type=int, default=16,
                    help='number of thread (default: 16)')

parser.add_argument('--window-size', type=int, default=8,
                    help='skip-gram window size (default: 8)')

parser.add_argument('--corpus_dir', type=str, default='data',
                    help='training resource dir')

parser.add_argument('--train', action='store_true', default=True,
                    help='do embedding trainig (default: True)')

parser.add_argument('--model-file', type=str, default='kospacing_wv.mdl',
                    help='output object from Word2Vec() (default: kospacing_wv.mdl)')

parser.add_argument('--numpy-wv', type=str, default='kospacing_wv.np',
                    help='numpy object file path from Word2Vec() (default: kospacing_wv.np)')

parser.add_argument('--w2idx', type=str, default='w2idx.dic',
                    help='item to index json dictionary (default: w2idx.dic)')

parser.add_argument('--model-dir', type=str, default='model',
                    help='dir to save models (default: model)')

opt = parser.parse_args()

if opt.train:
    # 이상하게 w2v 는 그대로고 mdl 파일 용량이 줄어든다. 그런데 완전히 새로 만드는 용량보다 크긴함.. 그리고 여러번 실행하면 mdl 용량이 늘어남.
    """
    total 9348
    drwxrwxr-x  2 seoul seoul      94  6월 27 18:40 ./
    drwxrwxr-x 10 seoul seoul     310  6월 27 15:50 ../
    -rw-rw-r--  1 seoul seoul 5426488  6월 27 14:47 kospacing.params
    -rw-rw-r--  1 seoul seoul 2512441  6월 27 18:40 kospacing_wv.mdl <- The full model file (사실 이것만 있으면 돼)
    -rw-rw-r--  1 seoul seoul 1595280  6월 27 18:40 kospacing_wv.np  <- The full model file 에서 파생된 vector file
    -rw-rw-r--  1 seoul seoul   30342  6월 27 18:40 w2idx.dic        <- The full model file 에서 파생된 word & index file

    total 8520
    drwxrwxr-x  2 seoul seoul      94  6월 27 18:40 ./
    drwxrwxr-x 10 seoul seoul     310  6월 27 15:50 ../
    -rw-rw-r--  1 seoul seoul 5426488  6월 27 14:47 kospacing.params
    -rw-rw-r--  1 seoul seoul 1665175  6월 27 18:46 kospacing_wv.mdl <- 이어서 실행했음에도 용량이 줄어들었다.
    -rw-rw-r--  1 seoul seoul 1595328  6월 27 18:46 kospacing_wv.np  <- 약간 늘어남
    -rw-rw-r--  1 seoul seoul   30342  6월 27 18:46 w2idx.dic

    total 8520
    drwxrwxr-x  2 seoul seoul      94  6월 27 18:40 ./
    drwxrwxr-x 10 seoul seoul     310  6월 27 15:50 ../
    -rw-rw-r--  1 seoul seoul 5426488  6월 27 14:47 kospacing.params
    -rw-rw-r--  1 seoul seoul 1665974  6월 27 18:48 kospacing_wv.mdl <- 한번 더 실행하니 늘어났다. 
    -rw-rw-r--  1 seoul seoul 1595328  6월 27 18:48 kospacing_wv.np  <- 같은 파일을 두번 실행하면 변화없음
    -rw-rw-r--  1 seoul seoul   30342  6월 27 18:48 w2idx.dic
    """
    update_full_model(opt.corpus_dir,
                        opt.model_dir + '/' + opt.model_file,
                        opt.model_dir + '/' + opt.numpy_wv,
                        opt.model_dir + '/' + opt.w2idx,
                        min_count=opt.min_count,
                        workers=opt.num_worker,
                        window=opt.window_size)

else:
    create_embeddings(opt.corpus_dir,
                        opt.model_dir + '/' + opt.model_file,
                        opt.model_dir + '/' + opt.numpy_wv,
                        opt.model_dir + '/' + opt.w2idx,
                        min_count=opt.min_count,
                        workers=opt.num_worker,
                        window=opt.window_size)
