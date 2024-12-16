import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a social network graph
def create_social_network():
    G = nx.Graph()

    # Add nodes (users) and edges (connections)
    G.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3),
        (2, 4), (3, 4), (3, 5), (4, 5)
    ])

    # Add features for each node (e.g., user interests or activity levels)
    for i in range(len(G.nodes)):
        G.nodes[i]['feature'] = np.random.rand(3)  # Random 3D features

    return G

# Step 2: Convert the graph to PyTorch Geometric format
def convert_to_pyg_graph(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    features = torch.tensor([G.nodes[i]['feature'] for i in G.nodes], dtype=torch.float)

    # Example labels: 0 for one community, 1 for another
    labels = torch.tensor([0 if i % 2 == 0 else 1 for i in G.nodes], dtype=torch.long)

    return Data(x=features, edge_index=edge_index, y=labels)

# Step 3: Define a GNN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 4: Train the GNN model
def train_gnn(data):
    # Split data into train and test
    train_mask, test_mask = train_test_split(range(len(data.y)), test_size=0.2, random_state=42)
    
    # Model, loss, and optimizer
    model = GCN(input_dim=data.x.shape[1], hidden_dim=16, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits[test_mask].argmax(dim=1)
        accuracy = accuracy_score(data.y[test_mask].cpu(), predictions.cpu())
        print(f"Test Accuracy: {accuracy:.2f}")

    return model

# Step 5: Predict friend recommendations or detect communities
def analyze_social_network(data, model):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        community_predictions = logits.argmax(dim=1).numpy()

    print("Community Predictions:")
    for i, community in enumerate(community_predictions):
        print(f"Node {i}: Community {community}")

# Main workflow
if __name__ == "__main__":
    # Create and process the social network graph
    G = create_social_network()
    data = convert_to_pyg_graph(G)

    # Train the GNN model
    model = train_gnn(data)

    # Analyze the social network
    analyze_social_network(data, model)
