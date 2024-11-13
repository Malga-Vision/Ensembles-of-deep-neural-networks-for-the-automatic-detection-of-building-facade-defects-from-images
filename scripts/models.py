import torch
import time
from torchvision import models
from torchvision.models import densenet201, resnet50, vgg16, vit_b_16, convnext_base, swin_b, swin_t
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os



def load_model(model_name, drop, num_classes, folder_path):
    if model_name == "densenet":
        base_model = densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        num_features = base_model.classifier.in_features
        base_model.classifier =  nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, int(num_classes)))

    elif model_name == "resnet":
        base_model = resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))

    elif model_name == "vgg":
        base_model = vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_features = base_model.classifier[0].in_features
        base_model.classifier = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))

    elif model_name == "vit":
        base_model = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        base_model.heads = nn.Sequential(
                    nn.Linear(768, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))
        
    elif model_name == 'convnext':
        base_model = convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        base_model.classifier = nn.Sequential(
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))
        
    elif model_name == 'swin':
        base_model = swin_b(weights=models.Swin_B_Weights.DEFAULT)
        base_model.head = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))
        
    elif model_name == 'swin_t':
        base_model = swin_t(weights=models.Swin_T_Weights.DEFAULT)
        base_model.head = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, num_classes))
    
    path = model_name + ".pt"
    path = os.path.join(folder_path, path)
    if not os.path.exists(path):
        torch.save(base_model, path)
    return base_model



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, epoch = 0, patience=10, verbose=True, delta=0.0001, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.best_val_acc = np.Inf
        self.best_train_acc = np.Inf
        self.epoch = epoch
        self.best_epoch = 0

    def __call__(self, val_loss, model, val_acc, train_acc, epoch):

        score = val_loss
        val_accuracy = val_acc
        train_accuracy = train_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, val_acc, train_acc, epoch)

        if score < self.best_score - self.delta:
          self.best_score = score
          self.best_val_acc = val_accuracy
          self.best_train_acc = train_accuracy
          self.best_epoch = epoch
          self.save_checkpoint(val_loss, model, val_acc, train_acc, epoch)
          self.counter = 0

        else:
            self.counter += 1
            if self.verbose:
              print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
    def save_checkpoint(self, val_loss, model, val_acc, train_acc, epoch):
        #Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.best_val_acc = val_acc
        self.best_train_acc = train_acc
        self.best_epoch = epoch






def training(epochs, model, train_loader, val_loader, model_file, lr, opt = 'adam', sched = 'exp', wd = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

    if sched == 'exp':
        lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1, verbose=True)
    if sched == 'plateau': 
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=4, verbose=True)

    H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    early_stopping = EarlyStopping(path = model_file)
    best_epoch = 0

    for epoch in range(epochs):
        print('#' * 50 + '  Epoch {}  '.format(epoch+1) + '#' * 50)
        train_start_time = time.time()
        start_time = time.time()

        ########################### Training phase  ###########################
        model.train()
        running_train_loss = 0.0
        running_corrects_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = running_corrects_train.double() / total_train
        
        train_epoch_duration = float(time.time() - train_start_time)

        ########################### Validation phase  #######################
        val_start_time = time.time()
        model.eval()
        running_val_loss = 0.0
        running_corrects_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)
        
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_corrects_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_acc = running_corrects_val.double() / total_val
        
        val_epoch_duration = float(time.time() - val_start_time)
        epoch_duration = float(time.time() - start_time)

        if sched == 'exp':
            lr_sched.step()
        else:
            lr_sched.step(epoch_val_loss)
        
        H["train_loss"].append(epoch_train_loss)
        H["train_acc"].append(epoch_train_acc.item())
        H["val_loss"].append(epoch_val_loss)
        H["val_acc"].append(epoch_val_acc.item())
        
        print('\n' + '-'*50 + f'  SUMMARY  ' + '-'*50)
        print(f'Training Phase.')
        print(f'  Total Duration:         {int(np.ceil(train_epoch_duration / 60)) :d} minutes')
        print(f'  Train Loss:     {epoch_train_loss :.3f}')
        print(f'  Train Accuracy: {epoch_train_acc :.3f}')
        
        print('Validation Phase.')
        print(f'  Total Duration:              {int(np.ceil(val_epoch_duration / 60)) :d} minutes')
        print(f'  Validation Loss:     {epoch_val_loss :.3f}')
        print(f'  Validation Accuracy: {epoch_val_acc :.3f}')

        print(f'  Total Duration:         {int(np.ceil(epoch_duration / 60)) :d} minutes')
        
        # early stopping
        early_stopping(epoch_val_loss, model, epoch_val_acc, epoch_train_acc, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            val_acc = early_stopping.best_val_acc
            train_acc = early_stopping.best_train_acc
            best_epoch = early_stopping.best_epoch
            best_val_loss = early_stopping.val_loss_min
            print("Best epoch: ", best_epoch)
            break

    return H, int(best_epoch), train_acc, val_acc, best_val_loss, epoch_duration


def evaluation_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    model = model.to(device)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    predictions = []

    target_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            predictions.append(preds)
            total_predictions += targets.size(0)

            correct_predictions += torch.sum(preds == targets.data).item()
            target_labels.extend(targets.tolist())
            pred_labels.extend(preds.tolist())

    accuracy = 100 * correct_predictions / total_predictions

    return accuracy, pred_labels, target_labels




class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)

    def forward(self, x):

        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, num_models, num_classes)
        
        return outputs


def evaluation_ensemble(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #print("Device: ", device)
    model = model.to(device)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    predictions = []

    target_labels = []
    pred_labels = []

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        #targets = targets.squeeze()
    
        outputs = model(inputs)
        #preds = outputs.argmax(1)
        _, preds = torch.max(outputs, 1)
        predictions.append(preds)

        total_predictions += list(targets.size())[0]

        correct_predictions += torch.sum(preds == targets.data).item()
        target_labels.extend(targets.tolist())
        pred_labels.extend(preds.tolist())


    accuracy = 100 * correct_predictions / total_predictions

    return accuracy, pred_labels, target_labels




#take as input logits
def average_ensemble_logits(test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_predictions = 0
    correct_predictions = 0
    target_labels = []
    pred_labels = []

    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device)
        inputs = inputs.squeeze(1)
        targets = targets.to(device)
        total_predictions += targets.size(0)

        average = torch.mean(inputs, dim=1)
        preds = average.argmax(1)
        correct_predictions += (preds == targets).sum().item()
        
        target_labels.extend(targets.tolist())
        pred_labels.extend(preds.tolist())
        

    accuracy = 100 * correct_predictions / total_predictions


    return accuracy, pred_labels, target_labels


class LearnableConvEnsemble(nn.Module):
    def __init__(self, n):
        super(LearnableConvEnsemble, self).__init__()
        self.num_models = n
        self.conv1d = nn.Conv1d(in_channels=self.num_models, out_channels=10, kernel_size=(1), stride=1, padding=0, bias=True)
        self.conv1d2 = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=(1), stride=1, padding=0, bias=False)

    def forward(self, outputs):
        outputs = outputs.squeeze(1)
        convolved_output = self.conv1d(outputs)
        convolved_output = self.conv1d2(convolved_output)

        convolved_output = convolved_output.squeeze(1)
        
        return convolved_output





