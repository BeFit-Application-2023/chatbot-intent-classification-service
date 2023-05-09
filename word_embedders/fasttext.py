# Importing all needed modules.
from .base import BaseWordEmbedder
from .errors import *
import numpy as np
import fasttext.util
import fasttext
import torch


class FastTextEmbedder(BaseWordEmbedder):
    def __init__(self,
                 vector_dimension : int,
                 tokenize_fun : "function",
                 max_length : int,
                 pad_token : str = "",
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
        '''
        # Initializing the super class.
        super(FastTextEmbedder, self).__init__(vector_dimension, tokenize_fun, max_length, pad_token, **kwargs)

        # Creation of the model version mapper.
        self.model_mapper = {
            "wiki" : "wiki.en.bin",
            "cc" : "cc.en.300.bin"
        }

        # Validation of the version.
        if "version" in kwargs:
            self.version = self.model_mapper[kwargs["version"]]
        else:
            self.version = "cc.en.300.bin"

        # Loading the FastText model.
        try:
            self.ft = fasttext.load_model(self.version)
        except:
            raise NotAValidVersion(f"{self.version} is missing! Download it!")

        # Validation of the vector dimension.
        if self.vector_dimension > 300 or self.vector_dimension < 0:
            raise ImpossibleDimension("Can't set vector dimensions more than 300 or less than 0!")
        elif self.vector_dimension != 300:
            self.ft = fasttext.util.reduce_model(self.ft, self.vector_dimension)

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
            if token in self.ft.words:
                embeds.append(self.ft.get_word_vector(token))
            else:
                embeds.append(np.zeros(self.vector_dimension, dtype=np.float32))

        # Converting the embeddings to Pytorch tensor and returning it.
        embeds = np.stack(embeds)
        return torch.from_numpy(embeds)