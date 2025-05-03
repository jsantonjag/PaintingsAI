# 1. Carga librerías
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

# 2. Transformaciones
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 3. Dataset personalizado
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
    
# Carga del dataset y de la carpeta de imágenes  
# Obtener la ruta absoluta del directorio donde está el script .py
script_dir = os.path.dirname(os.path.abspath("1001_images"))

# Ruta a la carpeta de imágenes que está en el mismo lugar que el script
img_dir = os.path.join(script_dir, "1001_images")
csv_path = "https://raw.githubusercontent.com/jsantonjag/PaintingsAI/refs/heads/main/data/dataset_completo.csv"

df = pd.read_csv(csv_path)
df['path'] = df['path'].apply(lambda x: os.path.join(img_dir, x))

# Mostrar algunas filas
df.head()

# 4. Cargar datos

full_dataset = ArtDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)

# Dividir en train/test pequeño
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 5. Definimos la red multitarea
class MultiTaskCNN(nn.Module):
    def __init__(self, num_authors, num_styles):
        super(MultiTaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(128 * 16 * 16, 512)
        
        self.author_head = nn.Linear(512, num_authors)
        self.style_head = nn.Linear(512, num_styles)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.shared_fc(x)
        author_output = self.author_head(x)
        style_output = self.style_head(x)
        return author_output, style_output

# 6. Inicialización
num_authors = len(full_dataset.authors)
num_styles = len(full_dataset.styles)

model = MultiTaskCNN(num_authors, num_styles)

criterion_author = nn.CrossEntropyLoss()
criterion_style = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Entrenamiento inicial
n_epochs = 5

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0

    for images, authors, styles in train_loader:
        optimizer.zero_grad()
        author_preds, style_preds = model(images)

        loss_author = criterion_author(author_preds, authors)
        loss_style = criterion_style(style_preds, styles)

        loss = loss_author + loss_style
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
# 8. Evaluación rápida
model.eval()
correct_author = 0
correct_style = 0
total = 0

with torch.no_grad():
    for images, authors, styles in test_loader:
        author_preds, style_preds = model(images)

        _, predicted_authors = torch.max(author_preds, 1)
        _, predicted_styles = torch.max(style_preds, 1)

        correct_author += (predicted_authors == authors).sum().item()
        correct_style += (predicted_styles == styles).sum().item()
        total += authors.size(0)

print(f'Author Accuracy: {100 * correct_author / total:.2f}%')
print(f'Style Accuracy: {100 * correct_style / total:.2f}%')

