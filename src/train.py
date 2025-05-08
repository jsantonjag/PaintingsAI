import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from dataset import ArtDataset
from model import MultiTaskResNet_m

# Hiperparámetros y configuración
BATCH_SIZE = 32
EPOCHS = 20
DROPOUT_PROB = 0.2
LOSS_FUNCTION = nn.CrossEntropyLoss()
ACTIVATION_FN = nn.ReLU()
BATCHNORM = True
DROPOUT = True

# Transformaciones de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

def train():
    # Cargar dataset
    url = "https://raw.githubusercontent.com/jsantonjag/PaintingsAI/refs/heads/main/data/dataset_completo.csv"
    dataset = ArtDataset(csv_file=url , img_dir='data/1001_images', transform=train_transform)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, _ = random_split(dataset, [train_size, val_size, test_size])

    # Cálculo de pesos inversos para el sampler
    author_labels = [dataset[i][1] for i in range(len(dataset))]
    author_counts = torch.bincount(torch.tensor(author_labels))
    class_weights = 1.0 / author_counts.float()
    weights = class_weights[torch.tensor(author_labels)]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Inicializar modelo
    model = MultiTaskResNet_m(
        num_authors=len(dataset.authors),
        num_styles=len(dataset.styles),
        BATCHNORM=BATCHNORM,
        ACTIVATION_FN=ACTIVATION_FN,
        DROPOUT=DROPOUT,
        DROPOUT_PROB=DROPOUT_PROB,
        loss_function=LOSS_FUNCTION,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, author_labels, style_labels in train_loader:
            optimizer.zero_grad()
            artist_out, style_out = model(images)
            loss_artist = LOSS_FUNCTION(artist_out, author_labels)
            loss_style = LOSS_FUNCTION(style_out, style_labels)
            loss = 0.7 * loss_artist + 0.3 * loss_style
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

if __name__ == "__main__":
    train()
