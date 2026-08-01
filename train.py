import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# مسارات الداتاسيت
data_dir = "./Gender Detection Tiny"

train_dir = os.path.join(data_dir, "Train")
val_dir = os.path.join(data_dir, "Validation")
test_dir = os.path.join(data_dir, "Test")


# تجهيز الصور
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# تحميل الداتاسيت
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# استخدام GPU إذا كان متوفرًا
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))


# تحميل مودل ResNet18 مدرب مسبقًا
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# تجميد الطبقات القديمة لتسريع التدريب
for parameter in model.parameters():
    parameter.requires_grad = False

# تعديل الطبقة الأخيرة لتصنيف Male / Female
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 5
best_val_accuracy = 0.0


# التدريب
for epoch in range(epochs):
    model.train()

    training_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_number, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        if batch_number % 50 == 0:
           print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_number}/{len(train_loader)}")

    training_accuracy = (
        100 * correct_predictions / total_predictions
    )


    # التحقق باستخدام Validation
    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            validation_total += labels.size(0)
            validation_correct += (predicted == labels).sum().item()

    validation_accuracy = (
        100 * validation_correct / validation_total
    )

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Accuracy: {training_accuracy:.2f}% | "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

    # حفظ أفضل مودل
    if validation_accuracy > best_val_accuracy:
        best_val_accuracy = validation_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes
            },
            "best_gender_model.pth"
        )

        print("Best model saved.")


# اختبار أفضل مودل
checkpoint = torch.load(
    "best_gender_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total

print(f"Final Test Accuracy: {test_accuracy:.2f}%")
print("Training completed successfully.")