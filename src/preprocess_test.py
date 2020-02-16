import os
import unittest
from preprocess import clean_text, tokenize_text, load_embedding, replace_token_with_index, pad_sequence


class tweet_test(unittest.TestCase):
    def setUp(self):
        self.text = "@BTS_twt: We met @torikelly @iambeckyg @ciara https://t.co/j7jXeTHc4A"
        return

    def test_clean(self):
        expected_result = " we met"
        result = clean_text(self.text)
        self.assertEqual(result, expected_result)

    def test_tokenizer(self):
        expected_result = ['met']
        result_1 = tokenize_text(clean_text(self.text))
        self.assertEqual(result_1, expected_result)

    def test_replace(self):
        path = os.getcwd() + "/glove_twitter/glove_twitter_200d_clean.txt"
        embeddingMap = load_embedding(path=path, max_length_dictionary=None)
        expected_result = [517]
        result_2 = replace_token_with_index(
            tokenize_text(clean_text(self.text)), embeddingMap
        )

        self.assertEqual(result_2, expected_result)

    def test_padsequence(self):
        path = os.getcwd() + "/glove_twitter/glove_twitter_200d_clean.txt"
        embeddingMap = load_embedding(path=path, max_length_dictionary=None)
        expected_result = [517, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result_3 = pad_sequence(replace_token_with_index(
            tokenize_text(clean_text(self.text)), embeddingMap)
        )

        self.assertEqual(result_3, expected_result)
        self.assertEqual(result_3, expected_result)
