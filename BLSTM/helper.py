
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
import torch.optim as optim
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix
import itertools
from scipy.fftpack import fft

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Global Hyperparameters
sequence_length = 100
input_size = 160
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 768
num_epochs = 30
learningRate = 0.003
subSampleInterval = 4
classes_str = ('bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal')
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
classes_str1 = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']


def statprinter():
    print("Sequence Length: ", sequence_length)
    print("Input Size: ", input_size)
    print("Hidden Size: ", hidden_size)
    print("Number of Layers: ", num_layers)
    print("Number of Classes: ", num_classes)
    print("batch_size: ", batch_size)
    print("Running for EPOCHS: ", num_epochs)
    print("Leaerning Rate: ", learningRate)
    print("Device: ", device)


def dataTransforms():
    toFloat = transforms.Compose(
        [
            transforms.Lambda(lambda x: x[0:16000]),
            transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)
        ])
    return toFloat


def trainProcess(toFloat):
    trainSet = NSynth(
        "/local/sandbox/nsynth/nsynth-train",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family_str"])

    return trainSet


def testProcess(toFloat):
    testSet = NSynth(
        "/local/sandbox/nsynth/nsynth-test",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family_str"])

    return testSet


def validProcess(toFloat):
    validSet = NSynth(
        "/local/sandbox/nsynth/nsynth-valid",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family_str"])

    input_dimension = 0

    return validSet, input_dimension


def getLoaders(trainData, testData, validData):
    trainLoader = data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    testLoader = data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True, drop_last=True)
    validLoader = data.DataLoader(dataset=validData, batch_size=batch_size, shuffle=True, drop_last=True)

    return trainLoader, testLoader, validLoader


def setCriteria(model):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    return criterion, optimizer


def trainModel(trainLoader, testLoader, model, criterion, optimizer):
    saved_losses = []
    validation_losses = []
    total_step = len(trainLoader)

    for epoch in range(num_epochs):

        for i, (sounds, family, qualities) in enumerate(trainLoader, 0):
            running_loss = 0.0
            validation_loss = 0.0
            actual_train = []
            pred_train = []
            correct_train = 1
            total_train = 1
            actual_valid = []
            pred_valid = []
            correct_valid = 1
            total_valid = 1

            sounds = sounds.reshape(-1, sequence_length, input_size).to(device)
            sounds = sounds.float()
            sounds = sounds.to(device)
            family = family.to(device)

            # Forward pass
            outputs = model(sounds)
            _, predicted = torch.max(outputs.data, 1)

            actual_train += family.cpu().numpy().tolist()
            pred_train += predicted.cpu().numpy().tolist()

            total_train += family.size(0)
            correct_train += (predicted == family).sum().item()

            loss = criterion(outputs, family)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))
                print('Training Accuracy : %d %%' % (100 * correct_train / total_train))

        saved_losses.append(loss.item())
        running_loss = 0.0

        with torch.no_grad():
            for sounds, family, qualities in testLoader:
                sounds = sounds.reshape(-1, sequence_length, input_size).to(device)
                family = family.to(device)

                sounds = sounds.float()

                outputs = model(sounds)
                _, predicted = torch.max(outputs.data, 1)

                actual_valid += family.cpu().numpy().tolist()
                pred_valid += predicted.cpu().numpy().tolist()

                total_valid += family.size(0)
                correct_valid += (predicted == family).sum().item()

                loss = criterion(outputs, family)
                validation_loss += loss.item()

        validation_losses.append(loss.item())
        print('Epoch ' + str(epoch + 1) + ', validation_loss: ' + str(loss.item()))
        print('Validation Accuracy : %d %%\n' % (100 * correct_valid / total_valid))
        validation_loss = 0.0

    print('Finished Training')
    return saved_losses, validation_losses


def plot(saved_losses, validation_losses, accuracy):
    fig, ax = plt.subplots()

    x = np.linspace(1, num_epochs, num_epochs)
    saved_losses = np.array(saved_losses)

    ax.set_title("Learning Rate: " + str(learningRate) + " | Accuracy: {:.3f}".format(accuracy) + "% | Batch: " + str(
        batch_size))  

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.plot(x, saved_losses, color='orange', marker=".", label='Training Loss')
    ax.plot(x, validation_losses, color='green', marker='.', label='Validation Loss')
    ax.legend()
    fig.savefig('BLSTM_Loss')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('BLSTM_confusion_matrix.png')


def testModel(testLoader, model, test_length):
    # net.eval()
    correct = 0
    total = 0
    i = 0
    tracker = {}
    tracker2 = {}
    checker = 0
    lowerprobab = {}
    actual = []
    pred = []
    accuracy = 0
    track = []
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for sounds, family, qualities in testLoader:
            sounds = sounds.reshape(-1, sequence_length, input_size).to(device)
            family = family.to(device)

            sounds = sounds.float()

            outputs = model(sounds)
            _, predicted = torch.max(outputs.data, 1)

            actual += family.cpu().numpy().tolist()
            pred += predicted.cpu().numpy().tolist()
            total += family.size(0)
            correct += (predicted == family).sum().item()

            for i in range(batch_size):

                label_item = family[i].item()
                predicted_item = predicted[i].item()
                sound_item = sounds[i][0].cpu().numpy()

                output_probab = outputs[i][predicted_item].item()

                probs = outputs[i]
                if predicted_item not in track:
                    track.append(predicted_item)

                    max = 0
                    actualIndex = 0
                    for i, probability in enumerate(probs):
                        if probability > max:
                            max = probability
                            actualIndex = i

                    min = 99999
                    Class = 0
                    j = 0
                    for probability in probs:
                        if probability != outputs[i][predicted_item]:
                            if (max - probability) < min:
                                min = max - probability
                                Class = j

                        j += 1

                    plt.figure()
                    plt.rcParams["axes.titlesize"] = 8
                    Actual_probab = probs[int(label_item)].item()
                    Nearest_probab = probs[Class].item()
                    plt.title(
                        "Actual class: {} | Nearest Class : {} | Actual Probab:{:.5f} | Nearest Probab:{:.5f}".format(
                            classes_str1[int(label_item)], classes_str1[Class], Actual_probab, Nearest_probab))
                    plt.plot(sounds[i][0].cpu().numpy())

                    plt.savefig('correctClass_related_Probab_' + str(classes_str[label_item]) + '.png')

                if (predicted_item in tracker and predicted_item in lowerprobab):
                    continue

                if ((predicted_item == label_item) and (predicted_item not in tracker)):
                    if (output_probab > 0.7):
                        tracker[predicted_item] = sounds[i][0].cpu().numpy()
                        plt.figure()
                        plt.rcParams["axes.titlesize"] = 10
                        plt.title("Predicted: {} | Actual : {} | Probab(Predicted): {:.5f}".format(
                            classes_str1[int(predicted_item)], classes_str1[int(label_item)], output_probab))
                        plt.plot(sounds[i][0].cpu().numpy(), c='green')
                        plt.savefig('correct_predict_highProbab_' + str(classes_str[label_item]) + '.png')

                if ((predicted_item == label_item) and (predicted_item not in lowerprobab)):
                    if (output_probab < 0.4):
                        lowerprobab[predicted_item] = sounds[i][0].cpu().numpy()
                        plt.figure()
                        plt.rcParams["axes.titlesize"] = 10
                        plt.title("Predicted: {} | Actual : {} | Probab(Predicted): {:.5f}".format(
                            classes_str[predicted_item], classes_str[label_item], output_probab))
                        plt.plot(sounds[i][0].cpu().numpy(), c='orange')
                        plt.savefig('correct_predict_lowProbab_' + str(classes_str[label_item]) + '.png')

    print('\nAccuracy of the network on the ' + str(test_length) + ' sound samples: %d %%\n' % (
        100 * correct / total))

    return correct, total, actual, pred


def perClassAccuracy(correct, total, testLoader, model):
    class_correct = list(0. for i in range(10))
    class_total = list(1. for i in range(10))

    with torch.no_grad():
        for sounds, family, qualities in testLoader:
            sounds = sounds.reshape(-1, sequence_length, input_size).to(device)
            family = family.to(device)

            sounds = sounds.float()

            outputs = model(sounds)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == family).squeeze()
            for i in range(10):
                label = family[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracies_y = []
    for i in range(10):
        accuracies_y.append(100 * class_correct[i] / class_total[i])
        print('Accuracy of %5s : %2d %%' % (
            classes[i], accuracies_y[i]))
    accuracies_y.append(100 * correct / total)


def printDataStats(trainData, testData, validData):
    train_length = len(trainData)
    valid_length = len(validData)
    test_length = len(testData)
    print('Training data Size: ', train_length)
    print('Validation data Size: ', valid_length)
    print('Testing data Size: ', test_length)

