# Importing all needed libraries.
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer
from allennlp.data import Token, Vocabulary, TokenIndexer, Tokenizer
from allennlp.data.fields import ListField, TextField
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from .base import BaseWordEmbedder


class ELMoEmbedder(BaseWordEmbedder):
    def __init__(self,
                 vector_dimension : int,
                 tokenize_fun : "function",
                 max_length : int,
                 pad_token : str = "<PAD>",
                 **kwargs) -> None:
        '''
            This function creates and sets up the ELMoEmbedder class.
                :param vector_dimension: int
                    The size of the vectors created by the ELMo embedder.
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
        super(ELMoEmbedder, self).__init__(vector_dimension, tokenize_fun, max_length, pad_token, **kwargs)

        # Creation of the token indexer and vocabulary.
        self.token_indexer = ELMoTokenCharactersIndexer()
        self.vocab = Vocabulary()

        # Defining the sources of the model.
        elmo_options_file = (
            "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json"
        )
        elmo_weight_file = (
            "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5"
        )

        # Creation of the ELMo Token Embedder.
        self.elmo_embedding = ElmoTokenEmbedder(
            options_file=elmo_options_file, weight_file=elmo_weight_file,
            projection_dim=vector_dimension,
        )

        # Setting the ELMo embedding to eval.
        self.elmo_embedding.eval()

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

        # Converting tokens to allennlp specific Token class.
        tokens = [Token(token) for token in tokens]

        # Creation of the token tensors.
        text_field = TextField(tokens, {"elmo_tokens":self.token_indexer})
        text_field.index(self.vocab)
        token_tensor = text_field.as_tensor(text_field.get_padding_lengths())

        # Creation of the ebedder.
        embedder = BasicTextFieldEmbedder(token_embedders={"elmo_tokens": self.elmo_embedding})

        # Creation of the embedded tokens.
        tensor_dict = text_field.batch_tensors([token_tensor])
        embedded_tokens = embedder(tensor_dict)

        return embedded_tokens[0].detach()