from preprocessing.prepare_data import create_sequences

import torch.nn as nn
import torch.optim as optim

class TSMixer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TSMixer, self).__init__()
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=1),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1),
                nn.ReLU()) for _ in range(num_layers)],
            nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.mixer(x)
        x = x.transpose(1, 2)
        return x[:, -1, :]
    

def model_training(values, X_train_shape, train_loader, lr, epochs, hidden_size, num_layers, output_size):
    # ------------------------------------ Model training

    input_size = X_train_shape
    hidden_size = hidden_size
    num_layers = num_layers
    output_size = output_size

    model = TSMixer(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


#------------------------------------ Training

    epochs = epochs
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    return model