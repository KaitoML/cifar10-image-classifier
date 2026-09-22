import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights
from model_resnet import ResNet18Transfer
from functions import train_step, test_step, eval_model, calculate_time
from config import Config
from timeit import default_timer as timer
from pathlib import Path
from torchinfo import summary

# To prevent accidentally overwriting the best params while experimenting with the code
experimenting = True

# Directory where trained params (in '.pth' format) are located
Path('saved_params').mkdir(exist_ok=True)

# Configuration
config = Config(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    best_params_path= 'saved_params/model_best_params.pth' if not experimenting else 'saved_params/model_experimental_params.pth',
    n_epochs=10,
    batch_size=32,
    scheduler_patience=3,
    early_stopping_patience=5,
    l2=1e-4,
    scheduler_factor=0.5,
    image_resize=(128, 128),
    trainable_layers_with_lr={'fc': 1e-3, 'layer4': 1e-3, 'layer3': 1e-5}
)

# To retrieve normalization values for transforms
weights = ResNet18_Weights.DEFAULT
mean_std = weights.transforms()

tf_train = transforms.Compose([
    transforms.Resize(config.image_resize), # ImageNet standard is 224x224, but in order to save time the sizes are changed
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std.mean, std=mean_std.std)
])

tf_test = transforms.Compose([
    transforms.Resize(config.image_resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_std.mean, std=mean_std.std)
])

train_data = datasets.CIFAR10(
    root='cifar_data',
    train=True,
    transform=tf_train,
    target_transform=None,
    download=True
)

test_data = datasets.CIFAR10(
    root='cifar_data',
    train=False,
    transform=tf_test,
    target_transform=None,
    download=True
)

loaders = {
    'train': DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=(config.device == 'cuda')
    ),

    'test': DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=(config.device == 'cuda')
    )
}

class_names = train_data.classes

# We'll need those in app.py file
with open('class_names.py', 'w') as f:
    f.write(f"class_names = {class_names!r}")

torch.manual_seed(42)
model = ResNet18Transfer(freeze=True, num_classes=10).to(config.device)

param_group = []
for layer_name, lr in config.trainable_layers_with_lr.items():
    layer = getattr(model.backbone, layer_name)

    for param in layer.parameters():
        param.requires_grad = True

    param_group.append({'params': layer.parameters(), 'lr': lr})


optimizer = optim.Adam(param_group, weight_decay=config.l2)

criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',
    patience=config.scheduler_patience,
    factor=config.scheduler_factor
)
accuracy = Accuracy(task='multiclass', num_classes=len(class_names)).to(config.device)

if __name__ == '__main__':

    summary = summary(model, input_size=(config.batch_size, 3, config.image_resize[0], config.image_resize[1]))
    print(summary)
    print()

    start = timer()

    best_loss = float('inf')
    epochs_no_improvement = 0


    for epoch in range(config.n_epochs):

        train_loss, train_acc = train_step(epoch, config.n_epochs, model, criterion, optimizer, accuracy, config.device, loaders)
        test_loss, test_acc = test_step(epoch, config.n_epochs, model, criterion, accuracy, config.device, loaders)

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), config.best_params_path)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch [{epoch+1}/{config.n_epochs}]. Best parameters saved at '{config.best_params_path}'")
                break

        print(f"Results for Epoch [{epoch+1}/{config.n_epochs}]: Train loss: {train_loss:.7f}, train accuracy: {train_acc:.4f} | Test loss: {test_loss:.7f}, test accuracy: {test_acc:.4f}\n")


    end = timer()
    time_sec = int(end - start)
    time_str = calculate_time(time_sec)
    print(f"\nTotal training time for model '{model.__class__.__name__}' with current configuration: [{time_str}]\n")

    model = ResNet18Transfer(freeze=True, num_classes=10).to(config.device)
    model.load_state_dict(torch.load(config.best_params_path, map_location=config.device))
    eval_summary = eval_model(model, criterion, accuracy, config.device, loaders, time_str)
    print(eval_summary)
