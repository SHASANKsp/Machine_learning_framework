#imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#defining the model architecture - current model has three hidden layer
class ProteinClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, num_classes, dropout_rate):
        super(ProteinClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3) 
        self.fc4 = nn.Linear(hidden_dim3, num_classes) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

#Creating a dataloader object
class ProteinDataset(Dataset):
    def __init__(self, df, label_column):
        self.embeddings = torch.tensor(df.drop(columns=[label_column]).values, dtype=torch.float32)
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

#model training with back propagation and optimizier
def train_model(model, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            inputs, labels = batch
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

#model testing
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    avg_loss = total_loss / len(data_loader)
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(all_labels), yticklabels=np.unique(all_labels))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




#model inputs

#Layer shape
input_dim = 1024
hidden_dim1 = 512
hidden_dim2 = 259
hidden_dim3 = 128

#Number of classes 
num_classes = 4

#Dropout rate
dropout_rate = 0.5

#initiating the model
model = ProteinClassifier(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, num_classes, dropout_rate)

#loss function
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#looking for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#column_name
label_column = "Species"  

#use the loaded data(df) 
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=42)

#creating a dataloader object
train_dataset = ProteinDataset(train_df, label_column)
test_dataset = ProteinDataset(test_df, label_column)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

#model training
train_model(model, train_loader,epochs=50)

#model evaluating
evaluate_model(model, test_loader)