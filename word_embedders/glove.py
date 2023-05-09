# Importing all needed modules.
from torchtext.vocab import GloVe
from .base import BaseWordEmbedder
from .errors import *

class GloVeEmbedder(BaseWordEmbedder):
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
                    :param version: str['42B', '840B', 'twitter.27B', '6B']
                        The version of the glove embedder.
        '''
        # Initializing the super class.
        super(GloVeEmbedder, self).__init__(vector_dimension, tokenize_fun, max_length, pad_token, **kwargs)
        self.tokenize_fun = tokenize_fun
        self.max_length = max_length
        self.pad_token = pad_token

        # Creation of the model version mapper.
        self.version_dimensions = {
            "42B" : [300],
            "840B" : [300],
            "twitter.27B" : [25, 50, 100, 200],
            "6B" : [50, 100, 200, 300]
        }

        # Validation of the version and vector dimension.
        if "version" in kwargs:
            if kwargs["version"] in self.version_dimensions:
                if vector_dimension in self.version_dimensions[kwargs["version"]]:
                    self.vector_dimension = vector_dimension
                else:
                    raise ImpossibleDimension(f"Glove(version={kwargs['version']}) doesn't support the dimesion {vector_dimension}")
                self.version = kwargs["version"]
                self.glove = GloVe(self.version, dim=self.vector_dimension)
            else:
                raise NotAValidVersion("This version of GloVe is not defined!")
        else:
            raise NotAValidVersion("No version provided")

    def get_vectors(self, document : str) -> "torch.Tensor":
        '''
            This function converts a document to a torch tensor of grade 2 (matrix).
                :param document: str
                    The document to be embedded.
                :return: torch.Tensor
                    The tensor representing the word embeddings for the document.
        '''
        # Getting the tokens.
        tokens = self.tokenize_fun(document.lower())

        # Normalizing the length of the tokens by padding or pruning.
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.pad_token] * (self.max_length - len(tokens))

        return self.glove.get_vecs_by_tokens(tokens)