#This file has the modified implementation of RESNET 18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
from torchsummary import summary
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import decomposition
from sklearn import manifold


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from resnetModel import *
import copy
import random
import time
train_accs = []
valid_accs = []
train_los = []
valid_los = []

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# Training function
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

    
def evaluate(model, iterator, criterion, device):   
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred= model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Testing function
def test():
    best_valid_loss = float('inf')
    for epoch in range(80):
        print(epoch)
        
        start_time = time.monotonic()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        scheduler.step()
        print("done")
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        train_los.append(train_loss)
        valid_los.append(valid_loss)
        end_time = time.monotonic()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


def plot_confusion_matrix(labels, pred_labels, classes):
    print("Ploting")
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels = classes)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
    plt.savefig('confusion_matrix.png')
 
def plot_accuracy():
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuarcy.png')
    plt.clf()  # clear the figure after saving the plot

def plot_loss():
    plt.plot(train_los, label='Training Loss')
    plt.plot(valid_los, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()  # clear the figure after saving the plot

def get_predictions(model, iterator, device):
    labels = []
    probs = []
    images = []

    model.eval()
    model.load_state_dict(torch.load('best_model.pt'))
    model.to(device)

    # evaluating on test set
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
    with torch.no_grad():

          for (x, y) in iterator:

              x = x.to(device)

              y_pred = model(x)

              y_prob = F.softmax(y_pred, dim=-1)

              images.append(x.cpu())
              labels.append(y.cpu())
              probs.append(y_prob.cpu())

          images = torch.cat(images, dim=0)
          labels = torch.cat(labels, dim=0)
          probs = torch.cat(probs, dim=0)
          print("predicted")
    return labels, probs
        
if __name__ == '__main__': 
    ROOT = '.data'
    train_data = datasets.CIFAR10(root = ROOT, 
                                train = True, 
                                download = True)
    # Compute means and standard deviations along the R,G,B channel
     
    train_transforms = transforms.Compose([
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(32, padding = 2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    test_transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])
    train_data = datasets.CIFAR10(ROOT, 
                                train = True, 
                                download = True, 
                                transform = train_transforms)

    test_data = datasets.CIFAR10(ROOT, 
                                train = False, 
                                download = True, 
                                transform = test_transforms)


    # Define the dataloaders for training and testing
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                            shuffle=False, num_workers=2)
    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, 
                                            [n_train_examples, n_valid_examples])
    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms


    train_iterator = data.DataLoader(train_data, batch_size=128, shuffle=True)

    valid_iterator = data.DataLoader(valid_data, batch_size=100, shuffle=True)

    test_iterator = data.DataLoader(test_data, batch_size=100, shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')


    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model
    layer_structure =[2,1,1,1]
    print('==> Building model with layer structure :',layer_structure)
    model = ResNet18(layer_structure).to(device)
    summary(model, (3, 32, 32))

   

    # Define hyperparameters
    learning_rate = 0.1

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    print("==> Learning Rate " ,learning_rate)
    print("==> Optimizer " ,optimizer)
    print("==> Scheduler " ,scheduler)
    
    test()
    plot_accuracy()
    plot_loss()

    print("done") 
    #Confusion Matrix
    labels, probs = get_predictions(model, test_iterator, device)
    pred_labels = torch.argmax(probs, 1)
    plot_confusion_matrix(labels, pred_labels, classes) 
                                  
    

 