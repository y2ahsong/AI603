import torch
import torch.nn as nn
from torch.autograd import grad

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(NeuralNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def u(model, t, x):
    tx = torch.cat([t, x], dim=1)
    u_pred = model(tx)
    return u_pred

def f(model, t, x):
    t.requires_grad = True
    x.requires_grad = True

    u_pred = u(model, t, x)
    u_t = grad(u_pred, t, torch.ones_like(u_pred), create_graph=True)[0]
    u_x = grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    f_pred = u_t + u_pred * u_x - (0.01 / torch.pi) * u_xx
    return f_pred

def train_burgers(Nf, Nu, device, num_epochs=5000, lr=1e-3):
    t_f = torch.rand(Nf, 1).to(device)
    x_f = torch.rand(Nf, 1).to(device)

    t_u = torch.rand(Nu, 1).to(device)
    x_u = torch.rand(Nu, 1).to(device)
    y_u = torch.sin(torch.pi * x_u)
    
    model = NeuralNet(input_dim=2, hidden_dim=20, output_dim=1, n_layers=9).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        f_pred = f(model, t_f, x_f)
        mse_f = torch.mean(f_pred**2)

        u_pred = u(model, t_u, x_u)
        mse_u = torch.mean((u_pred - y_u)**2)

        loss = mse_u + mse_f
        loss.backward()
        optimizer.step()
        
    t_test = torch.rand(100, 1).to(device)
    x_test = torch.rand(100, 1).to(device)
    y_test = torch.sin(torch.pi * x_test)

    y_pred = u(model, t_test, x_test)
    l2_error = torch.norm(y_test - y_pred) / torch.norm(y_test)
    return l2_error.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Nf = 2000
Nu_values = [20, 40, 60, 80, 100, 200]
results = {}

for Nu in Nu_values:
    print(f"Training with Nf={Nf}, Nu={Nu}")
    l2_error = train_burgers(Nf, Nu, device)
    results[Nu] = l2_error

print("\nResults:")
print("Nu\tL2 Error")
for Nu, error in results.items():
    print(f"{Nu}\t{error:.1e}")