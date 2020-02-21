import os
import unittest
from preprocess import Preprocessor

path = os.getcwd() + "/glove_twitter/glove_twitter_200d_clean.txt"
preprocessor = Preprocessor(path=path, max_length_dictionary=None)


class tweet_test(unittest.TestCase):
    def setUp(self):
        self.text = "@BTS_twt: We met @torikelly @iambeckyg @ciara https://t.co/j7jXeTHc4A"
        return

    def test_clean(self):
        expected_result = " we met"
        result = preprocessor.clean_text(self.text)
        self.assertEqual(result, expected_result)

    def test_tokenizer(self):
        expected_result = ['met']
        result_1 = preprocessor.tokenize_text(preprocessor.clean_text(self.text))
        self.assertEqual(result_1, expected_result)

    def test_replace(self):
        expected_result = [517]
        result_2 = preprocessor.replace_token_with_index(
            preprocessor.tokenize_text(preprocessor.clean_text(self.text)), preprocessor.embeddingMap
        )
        self.assertEqual(result_2, expected_result)

    def test_padsequence(self):
        expected_result = [517, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result_3 = preprocessor.pad_sequence(preprocessor.replace_token_with_index(
            preprocessor.tokenize_text(preprocessor.clean_text(self.text)), preprocessor.embeddingMap)
        )
        self.assertEqual(result_3, expected_result)
        self.assertEqual(result_3, expected_result)

    def test_pipeline(self):
        expected_result = [517, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result_4 = preprocessor.pipeline(self.text)
        self.assertEqual(result_4, expected_result)