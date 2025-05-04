# Caso base
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Dataset personalizado
class ArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.authors = sorted(self.data['artist'].unique())
        self.styles = sorted(self.data['style'].unique())
        self.author_to_idx = {author: idx for idx, author in enumerate(self.authors)}
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path'].replace('\\', '/')
        full_path = os.path.join(self.img_dir, os.path.basename(img_path))
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        author = self.author_to_idx[self.data.iloc[idx]['artist']]
        style = self.style_to_idx[self.data.iloc[idx]['style']]
        return image, author, style

# Modelo multitarea configurable
class MultiTaskResNet(nn.Module):
    def __init__(self, num_authors, num_styles, BATCHNORM, ACTIVATION_FN, DROPOUT, DROPOUT_PROB, LINK_FN, loss_function):
        super(MultiTaskResNet, self).__init__()
        
        self.loss_FN = loss_function
        self.link_FN = LINK_FN
        self.activation_FN = ACTIVATION_FN
        self.batchNorm = BATCHNORM
        self.dropout = DROPOUT
        self.dropout_prob = DROPOUT_PROB
        
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features

        layers = [nn.Flatten()]
        if self.batchNorm:
            layers.append(nn.BatchNorm1d(in_features))
            
        layers.append(self.wrap_activation(self.activation_FN))
        
        if self.dropout:
            layers.append(nn.Dropout(self.dropout_prob))
        
        layers.append(nn.Linear(in_features, in_features))
        self.shared_head = nn.Sequential(*layers)
        self.fc_artist = nn.Linear(in_features, num_authors)
        self.fc_style = nn.Linear(in_features, num_styles)

    def wrap_activation(self, ACTIVATION_FN):
        if ACTIVATION_FN == torch.relu:
            return nn.ReLU()
        elif ACTIVATION_FN == torch.tanh:
            return nn.Tanh()
        else:
            raise ValueError("Función de activación no soportada. Usa torch.relu o torch.tanh.")


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_head(x)
        artist_output = self.link_FN(self.fc_artist(x), dim=1)
        style_output = self.link_FN(self.fc_style(x), dim=1)
        return artist_output, style_output
    
    def compute_loss(self, artist_logits, artist_targets, style_logits, style_targets):
        return self.loss_FN(artist_logits, artist_targets) + self.loss_FN(style_logits, style_targets)
    
    
def test_model(model, epochs, train_loader, test_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_list, train_acc_artist_list, test_acc_artist_list = [], [], []
    train_acc_style_list, test_acc_style_list = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_artist, correct_style, total = 0, 0, 0

        for images, artist_labels, style_labels in train_loader:
            images, artist_labels, style_labels = images.to(device), artist_labels.to(device), style_labels.to(device)
            optimizer.zero_grad()
            artist_logits, style_logits = model(images)
            loss = model.compute_loss(artist_logits, artist_labels, style_logits, style_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            with torch.no_grad():
                artist_preds = torch.argmax(artist_logits, dim=1)
                style_preds = torch.argmax(style_logits, dim=1)
                correct_artist += (artist_preds == artist_labels).sum().item()
                correct_style += (style_preds == style_labels).sum().item()
                total += images.size(0)

        train_loss_list.append(total_loss / len(train_loader))
        train_acc_artist_list.append(correct_artist / total)
        train_acc_style_list.append(correct_style / total)

        # Evaluación en test
        model.eval()
        correct_artist_test, correct_style_test, total_test = 0, 0, 0
        with torch.no_grad():
            for images, artist_labels, style_labels in test_loader:
                images, artist_labels, style_labels = images.to(device), artist_labels.to(device), style_labels.to(device)
                artist_logits, style_logits = model(images)
                artist_preds = torch.argmax(artist_logits, dim=1)
                style_preds = torch.argmax(style_logits, dim=1)
                correct_artist_test += (artist_preds == artist_labels).sum().item()
                correct_style_test += (style_preds == style_labels).sum().item()
                total_test += images.size(0)

        test_acc_artist_list.append(correct_artist_test / total_test)
        test_acc_style_list.append(correct_style_test / total_test)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Train Acc Artist: {correct_artist / total:.2%} - Test Acc Artist: {correct_artist_test / total_test:.2%}")

    return train_loss_list, train_acc_artist_list, test_acc_artist_list, train_acc_style_list, test_acc_style_list

# Preparar datos
script_dir = os.path.dirname(os.path.abspath("1001_images"))
img_dir = os.path.join(script_dir, "1001_images")
csv_path = "https://raw.githubusercontent.com/jsantonjag/PaintingsAI/refs/heads/main/data/dataset_completo.csv"
df = pd.read_csv(csv_path)
df['path'] = df['path'].apply(lambda x: os.path.join(img_dir, x))
full_dataset = ArtDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Entrenamiento con la ReLU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskResNet(
    num_authors=len(full_dataset.authors), 
    num_styles=len(full_dataset.styles),
    BATCHNORM = True, 
    ACTIVATION_FN = torch.relu, 
    DROPOUT = True, 
    DROPOUT_PROB = 0.3, 
    LINK_FN = torch.softmax, 
    loss_function = nn.CrossEntropyLoss(),
)

losses, train_artist_acc, test_artist_acc, train_style_acc, test_style_acc = test_model(
    model=model,
    epochs=5,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device
)


