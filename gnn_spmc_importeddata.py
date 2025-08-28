import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Step 1: Load CSV Data
df = pd.read_csv("matrix_converter_output.csv")  # Replace with your actual file

# Step 2: Select relevant columns and normalize
features = df[['Iout', 'S1', 'S2']]
target = df[['Vout']]

scaler_x = StandardScaler()
x_scaled = scaler_x.fit_transform(features)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(target)

# Step 3: Convert to PyTorch tensors
x = torch.tensor(x_scaled, dtype=torch.float)
y = torch.tensor(y_scaled, dtype=torch.float)

# Step 4: Create a fully connected edge index
num_nodes = x.shape[0]
edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional

# Step 5: Create PyG data object
data = Data(x=x, edge_index=edge_index, y=y)

# Train-test split mask
train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=0)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True
data.train_mask = train_mask
data.test_mask = test_mask

# Step 6: Define GNN model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 8)
        self.lin = nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        out = self.lin(x)
        return out

model = GNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 7: Training loop
for epoch in range(201):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Step 8: Predict and plot
model.eval()
with torch.no_grad():
    predictions = model(data)

predicted_vout = scaler_y.inverse_transform(predictions[test_mask].numpy())
actual_vout = scaler_y.inverse_transform(data.y[test_mask].numpy())

from sklearn.metrics import mean_squared_error, r2_score

# Compute metrics
mse = mean_squared_error(actual_vout, predicted_vout)
r2 = r2_score(actual_vout, predicted_vout)

print(f"Waveform MSE (test): {mse:.4f}")
print(f"R² Score (Waveform): {r2:.3f}")


plt.figure(figsize=(10, 5))
plt.plot(actual_vout, label="Actual Vout")
plt.plot(predicted_vout, label="Predicted Vout", linestyle="--")
plt.title("GNN Output Voltage Prediction")
plt.xlabel("Sample Index")
plt.ylabel("Vout")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.plot(range(len(losses)), losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")

