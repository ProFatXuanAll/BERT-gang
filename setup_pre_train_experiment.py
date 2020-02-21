from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import bertgang

pre_train_experiment_no = 1

path = (os.path.dirname(os.path.abspath(__name__))
    + f'/data/pre_train_experiment/{pre_train_experiment_no}')

if os.path.exists(path):
    if not os.path.isdir(path):
        raise OSError(f'{path} is not a directory.')
else:
    os.makedirs(path)


config = bertgang.bertgang_config.PreTrainConfig(
    attention_probs_dropout_prob=0.0,
    batch_size=9,
    classifier_dropout_prob=0.1,
    do_lower_case=True,
    embedding_size=128,
    hidden_act='gelu_new',
    hidden_dropout_prob=0.0,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    inner_group_num=1,
    layer_norm_eps=1e-12,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_groups=1,
    num_hidden_layers=6,
    path=path,
    teachers=['bert', 'roberta', 'albert', ],
    type_vocab_size=2,
    vocab_size=30000
)

config.save_pretrained(config.path)

tokenizer = (bertgang
             .bertgang_tokenizer
             .Tokenizer
             .from_pretrained(bertgang
                              .bertgang_config
                              .VALID_TEACHERS['albert']['version']))
# TODO: This method does not work.
tokenizer.learn_from_teachers(config)
tokenizer.save_pretrained(config.path)