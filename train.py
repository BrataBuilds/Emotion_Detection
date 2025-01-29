"""
README
Iterations made over the previous versions:
1.Uploading the training data set took too long so made it self hosted.
2. Rewrote the program in torch.
3. Added the ability to record metrics such as training accuracy and testing accuracy over time
4. Added a checkpoint system.
5. Added a loading bar during training.
6. Added some basic fine tuning : 
- Increased batch-size
- Learning rate set to : 0.001
- Decay is set to 1e-6
- Using adam optimizer
- Using 4 layers.
- Regularization : Using two dropout zones.
- Downsampling Three MaxPooling zone.
- Basic Data-Augmentation
TODO: Make the model, metrics data downloadable locally automatically.
"""

import modal
import modal.runner
import modal.running_app
app = modal.App("Emotion-Recognition")
img = (modal.Image.debian_slim().pip_install("torch",
    "pillow",
    "numpy",
    "torchvision",
    "kagglehub")
    # .add_local_dir("data/test",remote_path="/root/test")
    # .add_local_dir("data/train",remote_path="/root/train")
    )
@app.function(image=img,gpu='t4',timeout=3600)
# Downloading the dataset on modal's servers cause uploading is slow.

def cnn():
    import kagglehub
    path = kagglehub.dataset_download("msambare/fer2013")
    # print("Path to dataset files:", path)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    import json
    from tqdm import tqdm, trange
# Defining the CNN architecture, more info given in README.md
    class EmotionCNN(nn.Module):
        def __init__(self):
            super(EmotionCNN, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )

            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout(0.35)

            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout(0.50)

            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.pool3 = nn.MaxPool2d(2, 2)


            self.flatten_size = 128 * 6 * 6  

            # Fully connected layers
            self.fc1 = nn.Linear(self.flatten_size, 1024)
            self.fc1_relu = nn.ReLU()
            self.dropout3 = nn.Dropout(0.50)
            self.fc2 = nn.Linear(1024, 7)
            self.softmax = nn.Softmax(dim=1)

        def forward_pass(self, x):
            # Data-flow visual in README.md
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.conv3(x)
            x = self.pool2(x)
            x = self.conv4(x)
            x = self.pool3(x)
            x = self.dropout2(x)
            
            # Flatten the output
            
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc1_relu(x)
            x = self.dropout3(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x

    # Data loading and preprocessing
    def create_data_loaders(train_dir, test_dir, batch_size=64):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((48, 48)),  
            
            # Basic Data Augmentation.
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
            transforms.ToTensor(),  
        ])

        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform
        )

        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=transform
        )
        
    # Data is Shuffled.
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, 

    def train_model(model, train_loader, test_loader, num_epochs=65, save_dir="model_outputs"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
        import os
        os.makedirs(save_dir, exist_ok=True)

        metric_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        # Epoch Progress Bar:
        epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', unit='epoch')

        for epoch in epoch_pbar:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} (Training)', leave=False, unit='batch', disable=True):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total

            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc='Validation', leave=False, unit='batch', disable=True):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100.0 * val_correct / val_total

            metric_history['epoch'].append(epoch + 1)
            metric_history['train_loss'].append(epoch_loss)
            metric_history['train_accuracy'].append(train_accuracy)
            metric_history['val_accuracy'].append(val_accuracy)

            epoch_pbar.set_postfix({
                'epoch': f'{epoch+1}/{num_epochs}',
                'loss': f'{epoch_loss:.4f}',
                'train_acc': f'{train_accuracy:.2f}%',
                'val_acc': f'{val_accuracy:.2f}%'
            })

            # Saving the model checkpoint for future analysis (PS: Didn't do that)
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, checkpoint_path)

        # Saving final model to use it during detection.
        final_model_path = os.path.join(save_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

        history_path = os.path.join(save_dir, "training_metrics.pth")
        torch.save(metric_history, history_path)

        print(f"Final model saved at {final_model_path}")
        print(f"Training metrics saved at {history_path}")

        return model, metric_history

    # Main execution

    if __name__ == "__main__":
        model = EmotionCNN().remote()
        train_loader, test_loader = create_data_loaders(
            train_dir="/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train",
            test_dir="/root/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test",
            batch_size=500
        ).remote()
        trained_model = train_model(model, train_loader, test_loader).remote()