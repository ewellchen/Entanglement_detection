# -*- coding: utf-8 -*-

"""Default configurations of model configuration, training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

MODEL_CONFIG = {

    'c1': 10,  'k1': 2,

    'c2' : 30, 'k2': 2,

    'd1' :96,
}

TRAIN_CONFIG = {
    'is_train' : True,

    'generate_data' : False,

    'train_size' : 160000,

    'test_size' : 40000,

    'train_set_path': './Data/sep_train_set.npy',

    's_test_set_path': './Data/sep_test_set.npy',

    'e_test_set_path': './Data/ent_test_set.npy',

    'result_path': 'result.txt',

    'train_config': {  'epoch': 100,
                        'batch_size': 1024,
                        'device': 'cuda:0', },

    'validation_data_config': { 'batch_size': 2000,
                                'device': 'cuda:0', },


}

