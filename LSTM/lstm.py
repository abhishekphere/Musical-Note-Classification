
import matplotlib
matplotlib.use('Agg')
import torch.utils.data as data
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from helper import dataTransforms, trainProcess, testProcess, validProcess, getLoaders, setCriteria,\
    trainModel, plot, plot_confusion_matrix, testModel, perClassAccuracy, printDataStats, statprinter, device

# Global Hyperparameters
sequence_length = 100
input_size = 160
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 768
num_epochs = 30
learningRate = 0.003
classes_str1 = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def main():
    # Prints hyperparameters
    print("================ Hyperparameters ====================")
    statprinter()

    # Gets tranforms
    transform = dataTransforms()

    # input data
    trainData = trainProcess(transform)
    testData = testProcess(transform)
    validData, input_dimension = validProcess(transform)

    # Load Data
    print("================LOADING DATA====================")
    trainLoader, testLoader, validLoader = getLoaders(trainData, testData, validData)

    printDataStats(trainData, testData, validData)

    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion, optimizer = setCriteria(model)

    print("================TRAINING MODEL====================")
    saved_losses, validation_losses = trainModel(trainLoader, testLoader, model, criterion, optimizer)
    # Save the model checkpoint
    torch.save(model.state_dict(), 'lstm_model.ckpt')

    print("================TEST MODEL====================")
    correct, total, actual, pred = testModel(testLoader, model, len(testData))

    accuracy = 100 * correct / total
    plot(saved_losses, validation_losses, accuracy)
    confusionmMat = confusion_matrix(actual, pred)
    print("...................................")
    print("Confusion Matrix:", confusionmMat)
    print("...................................")

    plt.figure()
    plot_confusion_matrix(confusionmMat, classes=classes_str1, normalize=True,
                          title='Normalized confusion matrix')

    perClassAccuracy(correct, total, testLoader, model)

if __name__ == '__main__':
    main()
