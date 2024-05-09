import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import torchvision
import torchvision.transforms as transforms
from six.moves import cPickle

def unpickle(f):
    datadict = cPickle.load(f,encoding='latin1')
    f.close()
    return datadict

def load_cifar10_dataset(dataset_dir):
    cifar10_data = []
    for filename in os.listdir(dataset_dir):
        with open(os.path.join(dataset_dir, filename), 'rb') as f:
            batch_data = unpickle(f)
            cifar10_data.extend(batch_data)
    return cifar10_data

# def load_cifar10_dataset(file_path):
#     with open(file_path, 'rb') as f:
#         cifar10_data = pickle.load(f, encoding='bytes')
#     return cifar10_data

def preprocess_cifar10_dataset(cifar10_data):
    images = cifar10_data[b'data']
    labels = cifar10_data[b'labels']
    
    # Normalize pixel values
    images = images / 255.0
    
    # Reshape images to 32x32x3
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
cifar10_data = load_cifar10_dataset('/Users/ar_khare/Downloads/Sem_2/Assignment/cifar-10-batches-py')
X_train, X_test, y_train, y_test = preprocess_cifar10_dataset(cifar10_data)
# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)