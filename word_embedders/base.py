class BaseWordEmbedder:
    def __init__(self,
                 vector_dimension : int,
                 tokenize_fun : "function",
                 max_length : int,
                 pad_token : str = "<PAD>",
                 **kwargs) -> None:
        '''
            This function creates and sets up the WordEmbedder class.
                :param vector_dimension: int
                    The size of the vectors created by the vector embedders.
                :param tokenize_fun: function
                    The torch or nltk tokenizing function used to tokenize documents.
                :param max_length: int
                    The number of tokens to pad the document to.
                :param pad_token: str, default = <PAD>
                    The string representing the pad token.
                :params kwargs:
                    Additional parameters for word emebedders.
        '''
        self.vector_dimension = vector_dimension
        self.tokenize_fun = tokenize_fun
        self.max_length = max_length
        self.pad_token = pad_token

    def get_vectors(self, document : str) -> "torch.Tensor":
        '''
            This function converts a document to a torch tensor of grade 2 (matrix).
                :param document: str
                    The document to be embedded.
        '''
        pass