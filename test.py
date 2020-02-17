from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import unittest

import create_pretrain_data

class TestCreatePretrainData(unittest.TestCase):
    def setUp(self):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = f'{self.current_path}/data'
        self.example_file = f'{self.data_path}/example.txt'

    def tearDown(self):
        pass

    def test_list_all_files(self):
        f = create_pretrain_data.list_all_files

        with self.assertRaises(TypeError):
            f(None)

        with self.assertRaises(FileNotFoundError):
            f(str(random.random()))

        with self.assertRaises(OSError):
            f(f'{self.data_path}/.gitignore')

        with self.assertRaises(TypeError):
            f(self.data_path, 'random text')

        with self.assertRaises(TypeError):
            f(self.data_path, ['random text', 123,])

        self.assertEqual(f(self.data_path),
                         ['.gitignore', 'example.txt',])

        self.assertEqual(f(self.data_path, ['a', 'b', 'c',]),
                         ['.gitignore', 'example.txt'])

        self.assertEqual(f(self.data_path, ['a', 'b', 'c', '.gitignore',]),
                         ['example.txt',])

    def test_sample_sentences_from_document(self):
        f = create_pretrain_data.sample_sentences_from_document

        with self.assertRaises(TypeError):
            f(None)

        with self.assertRaises(FileNotFoundError):
            f(str(random.random()))

        with self.assertRaises(OSError):
            f(self.data_path)

        with self.assertRaises(TypeError):
            f(self.example_file, 'random text')

        with self.assertRaises(ValueError):
            f(self.example_file, 0)

        with self.assertRaises(ValueError):
            f(self.example_file, -1)

        random.seed(264)
        seed_264_sample = [[
            'There is also one M-type star, magnitude 4.81 34 Boötis.',
            'It is of class gM0.'
        ]]
        self.assertEqual(f(self.example_file), seed_264_sample)

if __name__ == '__main__':
    unittest.main()
