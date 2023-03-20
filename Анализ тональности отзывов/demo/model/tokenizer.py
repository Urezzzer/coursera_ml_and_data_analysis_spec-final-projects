from sklearn.pipeline import TransformerMixin
import spacy
import pandas as pd


class Tokenizer(TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_series = []
        for text in X:
            doc = self.nlp(text)
            tokens = []
            for token in doc:
                if token.pos_ == 'NUM' or token.pos_ == 'PRON':
                    tokens.append(token.pos_)
                elif not token.like_num:
                    if token.lemma_ == 'll':
                        tokens.append('be')
                    if token.lemma_ == 've':
                        tokens.append('have')
                    else:
                        tokens.append(token.lemma_)
            tokens = ' '.join(tokens)
            new_series.append(tokens)
        return pd.Series(new_series)
