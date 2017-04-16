# -*- coding=utf-8 -*-

import os

tmp_file = os.listdir('tmp/')
for i in tmp_file:
    if i is not 'ch_model_weight' or 'model_weight' or 'en_model_weight':
        os.remove('tmp/' + i)



