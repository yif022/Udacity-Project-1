import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms as T, models

def choose_archt(arch):
    if arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_units = model.classifier.in_features       
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_units = model.classifier.in_features
    else:
        print('Choose Densenet or VGG')
    return model, input_units

# Build network
def build_network(model, input_units, output_units, lr, device):  
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_units, int(input_units/2))),
                                                  ('relu', nn.ReLU()),
                                                  ('drop', nn.Dropout(0.2)),
                                                  ('fc2', nn.Linear(int(input_units/2), output_units)),
                                                  ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device);    
    return model, criterion, optimizer

def train_model(model, dataloaders_train, dataloaders_valid, optimizer, criterion, lr, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 30
    for epoch in range(epochs):
        for images, labels in dataloaders_train:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            model.to(device)

            # Training forward pass, then backward pass, then update weights
            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                # set model to evaluation mode, dropout is off
                model.eval()
                with torch.no_grad():
                    # Validation pass
                    for images, labels in dataloaders_valid:
                        # Move input and label tensors to the default device
                        images, labels = images.to(device), labels.to(device)
                        model.to(device)
                        logps = model(images)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders_valid):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders_valid):.3f}")
                running_loss = 0
                # set model back to train mode
                model.train()
    return model

def test_model(model, dataloaders_test, criterion, device):    
    # TODO: Do validation on the test set
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders_test:
            images, labels = images.to(device), labels.to(device)
            model.to(device)
            logps = model(images)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(dataloaders_test):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloaders_test):.3f}")
    
    
def save_model(model, arch, image_datasets_train, optimizer, save_dir, epochs, input_units, output_units): 
    # TODO: Save the checkpoint 
    checkpoint = {'input_size': input_units,
                  'output_size': output_units,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'arch': arch,
                  'model.class_to_idx':image_datasets_train.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'epochs': epochs}

    torch.save(checkpoint, save_dir)

def load_checkpoint(model, save_dir):
    checkpoint = torch.load(save_dir)       
    model.load_state_dict = (checkpoint['state_dict'])
    model.classifier  = checkpoint['classifier']
    model.class_to_idx = checkpoint['model.class_to_idx']
    
    return model