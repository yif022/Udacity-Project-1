import argparse
import torch
from torch import nn
from torch import optim
from functions import build_network, train_model, test_model, save_model, choose_archt
from utility_func import load_preprocess_data



parser = argparse.ArgumentParser(description = 'Classify flower image') 

parser.add_argument('dir', type = str, default = 'flowers/', help = 'path to the folder of flower images') 
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth',  help = 'path to saving directory')
parser.add_argument('--arch', type = str, default = 'densenet161', help = 'choose densenet161 or densenet121 CNN architecture')
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'learning rate used')
parser.add_argument('--epochs', type = int, default = 20, help = 'epochs used')
parser.add_argument('--output_size', type = int, default = 102, help = 'outputs from the model')
parser.add_argument('--gpu', type = bool, default = False, help = 'use gpu')


args = parser.parse_args()

data_dir = args.dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
epochs = args.epochs
output_units = args.output_size
gpu = args.gpu

def main():
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Load and preprocess data 
    image_datasets_train, image_datasets_valid, image_datasets_test, dataloaders_train, dataloaders_valid, dataloaders_test = load_preprocess_data(data_dir)
    print('Data is loaded and preprocessed')

    # Build network and train model
    model, input_units = choose_archt(arch)
    model, criterion, optimizer = build_network(model, input_units, output_units, lr, device)   
    print('Model is training')
    model = train_model(model, dataloaders_train, dataloaders_valid, optimizer, criterion, lr, epochs, device)

    #Test model
    print('Model is evaluating')
    test_model(model, dataloaders_test, criterion, device)

    #Save model
    print('Saving Model')
    save_model(model, arch, image_datasets_train, optimizer, save_dir, epochs, input_units, output_units)
    print('Done.')  

if __name__=="__main__":
    main()