# Importing all needed modules.
import torch.nn as nn
import torch

# Defining the activation layer factory.
activation_layer_factory = {
    "relu" : nn.ReLU(),
    "tanh" : nn.Tanh(),
    "leaky_relu" : nn.LeakyReLU(),
    "selu" : nn.SELU(),
    "celu" : nn.CELU(),
    "gelu" : nn.GELU()
}


# Defining the Deep Learning model for text classification.
class LstmModel(nn.Module):
    def __init__(self, config_dict : dict) -> None:
        '''
            This function defines the architecture of the text classification
            neural network
                :param config_dict: dict
                    The configuration containing the configuration of the architecture
                    of the neural network.
        '''
        super(LstmModel, self).__init__()
        # Configuring the lstm layer.
        self.lstm_config = config_dict["lstm_config"]
        self.lstm = nn.LSTM(self.lstm_config["embedding_dim"],
                            self.lstm_config["lstm_hidden_dim"],
                            num_layers = self.lstm_config["lstm_num_layers"],
                            bidirectional = self.lstm_config["lstm_bidirectional"],
                            dropout = self.lstm_config["lstm_dropout"],
                            batch_first = True)
        # Extracting the linear config.
        self.linear_config = config_dict["linear_config"]

        # Creating an configuring the linear layers of the network.
        self.fully_conected_layers = nn.ModuleList()
        next_input_dim = self.lstm_config["lstm_hidden_dim"] * 2
        for i in range(len(self.linear_config)):
            self.fully_conected_layers.extend(
                nn.ModuleList(
                    [
                        nn.Linear(next_input_dim, self.linear_config[i]["output_dim"]),
                        nn.Dropout(self.linear_config[i]["dropout"]) if self.linear_config[i]["dropout"] else nn.Identity(),
                        nn.BatchNorm1d(self.linear_config[i]["output_dim"]) if self.linear_config[i]["batch_norm"] else nn.Identity(),
                        activation_layer_factory[self.linear_config[i]["activation"]] if self.linear_config[i]["activation"] else nn.Identity()
                    ]
                )
            )
            next_input_dim = self.linear_config[i]["output_dim"]

    def forward(self, embeds : "torch.Tensor"):
        '''
            This function forwards the inputs through the network layers.
                :param embeds: torch.Tensor
                    The embedding from the Word Embedding technique.
        '''
        # Passing the embedding through the lstm layer.
        packed_output, (hidden, cell) = self.lstm(embeds)

        # Concatenating the last outputs of the LSTM module.
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # Passing the output through hte linear layer.
        for i in range(len(self.fully_conected_layers)):
            out = self.fully_conected_layers[i](out)

        return out