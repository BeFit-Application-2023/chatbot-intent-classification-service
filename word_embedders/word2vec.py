# Importing all needed modules.
from gensim.models import KeyedVectors, Word2Vec
from .base import BaseWordEmbedder
import gensim.downloader as api
from .errors import *
import numpy as np
import torch

class Word2VecEmbedder(BaseWordEmbedder):
    def __init__(self,
                 vector_dimension : int,
                 tokenize_fun : "function",
                 max_length : int,
                 pad_token : str = "<PAD>",
                 **kwargs) -> None:
        '''
            This function creates and sets up the FastTextEmbedder class.
                :param vector_dimension: int
                    The size of the vectors created by the FastText embedder.
                :param tokenize_fun: function
                    The torch or nltk tokenizing function used to tokenize documents.
                :param max_length: int
                    The number of tokens to pad the document to.
                :param pad_token: str, default = <PAD>
                    The string representing the pad token.
                :params kwargs:
                    Additional parameters for word emebedders.
                    :param version: str['google-news-300']
                        The version of the glove embedder.
        '''
        # Initializing the super class.
        super(Word2VecEmbedder, self).__init__(vector_dimension, tokenize_fun, max_length, pad_token, **kwargs)

        # Creation of the model version mapper.
        self.model_mapper = {
            "google-news-300" : "word2vec-google-news-300"
        }

        # Validation of the version.
        if "version" in kwargs:
            if kwargs["version"] in self.model_mapper:
                self.version = kwargs["version"]
            else:
                raise NotAValidVersion(f"{kwargs['version']} is not a valid version for Word2Vec!")
        else:
            self.version = "google-news-300"

        # Loading of the Word2Vec model.
        self.model_version = self.model_mapper[self.version]
        self.w2v = api.load(self.model_version)

    def get_vectors(self, document : str) -> "torch.Tensor":
        '''
            This function converts a document to a torch tensor of grade 2 (matrix).
                :param document: str
                    The document to be embedded.
                :return: torch.Tensor
                    The tensor representing the word embeddings for the document.
        '''
        # Getting the tokens.
        tokens = self.tokenize_fun(document)

        # Normalizing the length of the tokens by padding or pruning.
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))

        # Creating and filling the embedding list.
        embeds = []
        for token in tokens:
            if self.w2v.has_index_for(token):
                embeds.append(self.w2v.get_vector(token))
            else:
                embeds.append(np.zeros(self.vector_dimension, dtype=np.float32))

        # Converting the embeddings to Pytorch tensor and returning it.
        embeds = np.stack(embeds)
        return torch.from_numpy(embeds)
