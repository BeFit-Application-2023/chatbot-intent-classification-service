# Importing all needed modules.
from nltk.tokenize import word_tokenize, wordpunct_tokenize, casual_tokenize
from nltk.tokenize.nist import NISTTokenizer
from torchtext.data.utils import get_tokenizer
from .word2vec import Word2VecEmbedder
from .elmo import ELMoEmbedder
from .glove import GloVeEmbedder
from .fasttext import FastTextEmbedder
from .errors import *


class WordEmbedderFactory:
    def __init__(self):
        '''
            The constructor of the word embedder factory.
            Doesn't do anything.
        '''
        pass

    def create_tokenizer(self, tokenizer : str) -> "function":
        '''
            This function returns a tokenization function by it's name.
            Now are supported the following list of tokenization functions:
                - torch.basic_english
                - nltk.word_tokenizer
                - nltk.casual_tokenizer
                - nltk.wordpunct_tokenizer
                - nltk.nist_tokenizer
            If the tokenizer is not in this list then an Exception is raised.
                :param tokenizer: str
                    The name of the tokenization function to be created.
                :return: function
                    The tokenization function. callable.
        '''
        if tokenizer == "torch.basic_english":
            return get_tokenizer("basic_english")
        elif tokenizer == "nltk.word_tokenizer":
            return word_tokenize
        elif tokenizer == "nltk.casual_tokenizer":
            return casual_tokenize
        elif tokenizer == "nltk.wordpunct_tokenizer":
            return wordpunct_tokenize
        elif tokenizer == "nltk.nist_tokenizer":
            return NISTTokenizer().tokenize
        else:
            raise Exception(f"{tokenizer} is not registered as a valid tokenizer!")

    def get_word_embedding(self, word_embed_config : dict) -> "WordEmbedder":
        '''
            This function creates word embedder by its configurations.
                :param word_embed_config: dict
                    Dictionary with the configurations of the word embedding method.
        '''
        # Creation of a copy of the word embedding configurations.
        self.config = word_embed_config.copy()

        # Extracting the word embedding method from configurations.
        word_embed_method = self.config["method"]
        del self.config["method"]

        # Creation of the tokenization function.
        self.config["tokenize_fun"] = self.create_tokenizer(self.config["tokenize_fun"])

        # Creation of the word embedder.
        if word_embed_method == "word2vec":
            return Word2VecEmbedder(**self.config)
        elif word_embed_method == "fasttext":
            return FastTextEmbedder(**self.config)
        elif word_embed_method == "elmo":
            return ELMoEmbedder(**self.config)
        elif word_embed_method == "glove":
            return GloVeEmbedder(**self.config)
        else:
            raise Exception(f"{word_embed_method} is not recognized!")
