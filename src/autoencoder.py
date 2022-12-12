import torch
import torch.nn as nn
import src.trainer_utils as utils
from collections import OrderedDict

device = utils.get_default_device()

# Autoencoder code without layer tuning. This example is written for 3 layers.
# class AutoEncoder(nn.Module):
#     def __init__(self, in_size, latent_size):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(in_size, int(in_size / 2)),
#             nn.ReLU(),
#             nn.Linear(int(in_size / 2), int(in_size / 4)),
#             nn.ReLU(),
#             nn.Linear(int(in_size / 4), latent_size),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_size, int(in_size / 4)),
#             nn.ReLU(),
#             nn.Linear(int(in_size / 4), int(in_size / 2)),
#             nn.ReLU(),
#             nn.Linear(int(in_size / 2), in_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input_window):
#         latent_window = self.encoder(input_window)
#         reconstructed_window = self.decoder(latent_window)
#         return reconstructed_window

# Autoencoder code with layer tuning.
class AutoEncoder(nn.Module):
    def __init__(self, in_size, latent_size, num_layers):
        super().__init__()

        num_neurons=[]
        for l in range(num_layers):
            num_neurons.append(in_size)
            in_size=int(in_size/2)
        num_neurons.append(latent_size)

        encoder_layers = OrderedDict()
        for layer_n in range(num_layers):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n+1])
            encoder_layers['layer_'+str(layer_n)] = h_layer
            encoder_layers['relu_'+str(layer_n)] = nn.ReLU()

        decoder_layers = OrderedDict()
        for layer_n in range(num_layers,0,-1):
            h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n-1])
            decoder_layers['layer_' + str(layer_n)] = h_layer
            if layer_n == 1:
                decoder_layers['sigmoid'] = nn.Sigmoid()
            else:
                decoder_layers['relu_' + str(layer_n)] = nn.ReLU()

        self.encoder = nn.Sequential(encoder_layers)
        self.decoder = nn.Sequential(decoder_layers)

    def forward(self, input_window):
        latent_window = self.encoder(input_window)
        reconstructed_window = self.decoder(latent_window)
        return reconstructed_window

def training(epochs, autoencoder_model, train_loader, learning_rate):
    history = []
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = utils.to_device(batch, device)
            recon = autoencoder_model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss: {loss.item():.4f}')
        history.append((epoch, batch, recon))
    return history

def testing(autoencoder_model, test_loader):
    results = []
    for [batch] in test_loader:
        batch = utils.to_device(batch, device)
        with torch.no_grad():
            recon = autoencoder_model(batch)
        results.append(torch.mean((batch - recon) ** 2, axis=1))
    return results