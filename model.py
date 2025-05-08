import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
from dataset import ArtDataset


# Transformaciones
transform_ = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
class MultiTaskResNet_m(nn.Module):
    def __init__(self, num_authors, num_styles, BATCHNORM, ACTIVATION_FN, DROPOUT, DROPOUT_PROB, loss_function):
        super(MultiTaskResNet_m, self).__init__()
        
        self.loss_FN = loss_function
        self.link_FN = lambda x, dim: x # No devuelve nada porque la softmax está implementada en la CrossEntropyLoss()
        self.activation_FN = ACTIVATION_FN
        self.batchNorm = BATCHNORM
        self.dropout = DROPOUT
        self.dropout_prob = DROPOUT_PROB
        
        base_model = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features

        layers = [nn.Flatten()]
        if self.batchNorm:
            layers.append(nn.BatchNorm1d(in_features))
        
        layers.append(nn.Linear(in_features, in_features))
        layers.append(self.wrap_activation(self.activation_FN))
        
        if self.dropout:
            layers.append(nn.Dropout(self.dropout_prob))
        
        layers.append(nn.Linear(in_features, in_features))
        layers.append(self.wrap_activation(self.activation_FN))
        
        self.shared_head = nn.Sequential(*layers)
        self.fc_artist = nn.Linear(in_features, num_authors)
        self.artist_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.style_head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
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
        
        artist_output = self.link_FN(self.fc_artist(self.artist_head(x)), dim=1)
        style_output = self.link_FN(self.fc_style(self.style_head(x)), dim=1)
        
        return artist_output, style_output
    
    def compute_loss(self, artist_logits, artist_targets, style_logits, style_targets):
        # ponderación opcional si una tarea es más difícil que otra
        return 0.7 * self.loss_FN(artist_logits, artist_targets) + 0.3 * self.loss_FN(style_logits, style_targets)
    
def test_model(model, epochs, train_loader, test_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
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

        scheduler.step(correct_artist_test/total_test)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Train Acc Artist: {correct_artist / total:.2%} - Train Acc Style: {correct_style / total:.2%}")

    return train_loss_list, train_acc_artist_list, test_acc_artist_list, train_acc_style_list, test_acc_style_list

    

# Preparar datos
script_dir = os.path.dirname(os.path.abspath("1001_images"))
img_dir = os.path.join(script_dir, "1001_images")
csv_path = "https://raw.githubusercontent.com/jsantonjag/PaintingsAI/refs/heads/main/data/dataset_completo.csv"
df = pd.read_csv(csv_path)

#Agrupar artistas con <10 imágenes como "Otros"
artist_counts = df['artist'].value_counts()
valid_artists = artist_counts[artist_counts >= 10].index.tolist()
df['artist'] = df['artist'].apply(lambda x: x if x in valid_artists else 'Otros')

df.to_csv("filtered_dataset.csv", index=False)

df['path'] = df['path'].apply(lambda x: os.path.join(img_dir, x))

full_dataset = ArtDataset(csv_file="filtered_dataset.csv", img_dir=img_dir, transform=transform_)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Calcular pesos inversos por clase
author_labels = [sample[1] for sample in train_dataset]
class_sample_counts = pd.Series(author_labels).value_counts().sort_index()
weights = 1. / class_sample_counts
sample_weights = [weights[label] for label in author_labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16)


#Entrenamiento con la ReLU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskResNet_m(
    num_authors=len(full_dataset.authors), 
    num_styles=len(full_dataset.styles),
    BATCHNORM = True, 
    ACTIVATION_FN = torch.relu, 
    DROPOUT = True, 
    DROPOUT_PROB = 0.2,  
    loss_function = nn.CrossEntropyLoss(),
)

losses_n, train_artist_acc_n, test_artist_acc_n, train_style_acc_n, test_style_acc_n = test_model(
    model=model,
    epochs=20,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device
)

fig, (ax11,ax21,ax31) = plt.subplots(1,3, figsize = (20,5))

## Plot cross entropy loss
#ax11.plot(range(len(losses)), losses, label = 'ResNet18 - loss', color = 'C0')
ax11.plot(range(len(losses_n)), losses_n, label = 'ResNet34 new - loss', color = 'C1')
ax11.set_xlabel('epochs')
ax11.legend()

## Plot train acc
#ax21.plot(range(len(train_artist_acc)), train_artist_acc, label = 'ResNet18 - artist', color = 'C0')
ax21.plot(range(len(train_artist_acc_n)), train_artist_acc_n, label = 'ResNet34 new - artist', color = 'C1')
ax21.set_ylabel('Train accuracy')
ax21.set_xlabel('epochs')
ax21.legend()

## Plot test acc
#ax31.plot(range(len(test_artist_acc)), test_artist_acc, label = 'ResNet18 - artist', color = 'C0')
ax31.plot(range(len(test_artist_acc_n)), test_artist_acc_n, label = 'ResNet34 new - artist', color = 'C1')
ax31.set_ylabel('Test accuracy')
ax31.set_xlabel('epochs')
ax31.legend()