from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import transformers


class Model(transformers.AlbertModel):

    def __init__(self, config):
        super().__init__(config)
