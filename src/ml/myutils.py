import nltk
from sklearn.base import TransformerMixin


PREFIX_WORD_NGRAM="W:"
PREFIX_CHAR_NGRAM="C:"


def get_size_tuple(ngram_str):
    """
    Convert n-gram string to tuple
    :param ngram_str:  "1-3" (lower and upper bound separated by hyphen)
    :return: tuple
    >>> get_size_tuple("3-5")
    (3, 5)
    >>> get_size_tuple("1")
    (1, 1)
    """
    if "-" in ngram_str:
        lower, upper = ngram_str.split("-")
        lower = int(lower)
        upper = int(upper)
    else:
        lower = int(ngram_str)
        upper = lower
    return (lower, upper)


class Featurizer(TransformerMixin):
    """Our own featurizer: extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        for all tweets of a user
        """
        out= [self._ngrams(text) for text in X]
        return out

    def __init__(self,word_ngrams="1",char_ngrams="0",binary=True):
        """
        binary: whether to use 1/0 values or counts
        lowercase: convert text to lowercase
        remove_stopwords: True/False
        """
        self.data = [] # will hold data (list of dictionaries, one for every instance)
        self.binary=binary
        self.word_ngram_size = get_size_tuple(word_ngrams)
        self.char_ngram_size = get_size_tuple(char_ngrams)

    def _ngrams(self,text):
        """
        extracts word or char n-grams

        range defines lower and upper n-gram size

        >>> f=Featurizer(word_ngrams="1-3")
        >>> d = f._ngrams("this is a test")
        >>> len(d)
        9
        >>> f=Featurizer(word_ngrams="0", char_ngrams="2-4")
        >>> d2 = f._ngrams("this")
        >>> len(d2)
        6
        """

        d={} # new dictionary that holds features for current instance

        lower, upper = self.word_ngram_size
        if lower != 0:
            for n in range(lower,upper+1):
                ## word n-grams
                for gram in nltk.ngrams(text.split(" "), n):
                    gram = "{}_{}".format(PREFIX_WORD_NGRAM, "_".join(gram))
                    if self.binary:
                        d[gram] = 1 #binary
                    else:
                        d[gram] = d.get(gram,0)+1

        c_lower, c_upper = self.char_ngram_size
        if c_lower != 0:
            for n in range(c_lower, c_upper + 1):
                ## char n-grams
                for gram in nltk.ngrams(text, n):
                    gram = "{}_{}".format(PREFIX_CHAR_NGRAM, "_".join(gram))
                    if self.binary:
                        d[gram] = 1  # binary
                    else:
                        d[gram] = d.get(gram, 0) + 1

        return d


if __name__ == "__main__":
    import doctest
    doctest.testmod()