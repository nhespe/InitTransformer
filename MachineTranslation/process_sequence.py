""" Data preprocessing for sequence to sequence texts
"""
from typing import List

import spacy
import pandas as pd

from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, TabularDataset

class SequenceTokenizer:
    def __init__(self, lang_a, lang_b):
        # Load Tokenizers
        self.a_tokenizer = spacy.load(lang_a)
        self.b_tokenizer = spacy.load(lang_b)

    def tokenize_sequence_pair(self, split_text_a: List[str], split_text_b: List[str]):
        """ Takes two inputs, returns iteratos of the  """
        ## Defines tokenizers
        def tokenize_a(sentence):
            return [tok.text for tok in self.a_tokenizer.tokenizer(sentence)]

        def tokenize_b(sentence):
            return [tok.text for tok in self.b_tokenizer.tokenizer(sentence)]

        A_TEXT = Field(tokenize=tokenize_a)
        B_TEXT = Field(tokenize=tokenize_b, init_token = "<sos>", eos_token = "<eos>")

        ## Applies the tokenizers
        df = _format_to_pandas(split_text_a, split_text_b)
        _split_and_persist(df)

        train, test = torchtext.data.TabularDataset.splits(
            path='/tmp/', train='train.csv', validation='test.csv', format='csv', 
            fields=[('lang_a', A_TEXT), ('lang_b', B_TEXT)]
        )

        A_TEXT.build_vocab(train, test)
        B_TEXT.build_vocab(train, test)

        # Turn this  into an iterator
        train_iter = BucketIterator(
            train, batch_size=20, sort_key=lambda x: len(x.lang_b), shuffle=True
        )

    def _format_to_pandas(data_a: List[str], data_b: List[str], trim_length=50) -> pd.DataFrame:
        """ Converts inputs to pandas, filters by length and compared-length """
        raw_data = {'lang_a' : [line for line in data_a], 'lang_b': [line for line in data_b]}
        df = pd.DataFrame(raw_data, columns=["lang_a", "lang_b"])

        df['lang_a_len'] = df['lang_a'].str.count(' ')
        df['lang_b_len'] = df['lang_b'].str.count(' ')

        # Remove pairs that are too long
        df = df.query(f'lang_b_len < {trim_length} & lang_a_len < {trim_length}')
        # Remove those that dont have equal lengths (allow for 1.25 diff in len)
        df = df.query('lang_b_len < lang_a_len * 1.25 & lang_b_len * 1.25 > lang_a_len')

        return df

    def _split_and_persist(df: DataFrame) -> None:
        """ train/test split on the df, then persists to train/test """
        train, test = train_test_split(df, test_size=0.1)
        train.to_csv("/tmp/train.csv", index=False)
        test.to_csv("/tmp/test.csv", index=False)
