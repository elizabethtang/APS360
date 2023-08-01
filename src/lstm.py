import torch
import torch.nn as nn
import torch.optim as optim

class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(lstm_model, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def train_lstm(model, device, train_loader, num_epochs=40, lr=0.0003):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #train the model
    num_epochs = num_epochs
    loss_curve = []
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(inputs)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                loss_curve.append(loss.item())
            
            model_path = "models/" + get_model_name(model.name, train_loader.batch_size, lr, epoch)
        torch.save(model.state_dict(), model_path)
    
    return loss_curve

def get_model_name(name, batch_size, learning_rate, epoch):
    return  "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
