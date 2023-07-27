import unittest
from .textual_inversions import extract_tokens_from_list


class TextualInversionsTest(unittest.TestCase):
    def test_extract_tokens_query_fname(self):
        tis = ["https://civitai.com/api/download/models/106132#fname=4nj0lie.pt"]
        tokens = extract_tokens_from_list(tis)
        self.assertEqual(tokens[0], "4nj0lie")

    def test_extract_tokens_query_token(self):
        tis = [
            "https://civitai.com/api/download/models/106132#fname=4nj0lie.pt&token=4nj0lie"
        ]
        tokens = extract_tokens_from_list(tis)
        self.assertEqual(tokens[0], "4nj0lie")
