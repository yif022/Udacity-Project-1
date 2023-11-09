import torch
from PIL import Image
from torchvision import datasets, transforms as T

#Load and preprocess the image dataset
def load_preprocess_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms_train = T.Compose([T.Resize(256),
                                       T.CenterCrop(224),
                                       T.RandomHorizontalFlip(),
                                       T.RandomVerticalFlip(),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])
    data_transforms_valid = T.Compose([T.Resize(256),
                                       T.CenterCrop(224),
                                       T.ToTensor(),
                                       T.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])

    data_transforms_test = T.Compose([T.Resize(256),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets_train = datasets.ImageFolder(data_dir + '/train', transform = data_transforms_train)
    image_datasets_valid = datasets.ImageFolder(data_dir + '/valid', transform = data_transforms_valid)
    image_datasets_test = datasets.ImageFolder(data_dir + '/test', transform = data_transforms_test)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size = 64, shuffle = True)
    dataloaders_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size = 64)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size = 64)

    return image_datasets_train, image_datasets_valid, image_datasets_test, dataloaders_train, dataloaders_valid, dataloaders_test


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im_transform = T.Compose([T.Resize(256),
                                 T.CenterCrop(224),
                                 T.ToTensor(),
                                 T.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    im_tensor = im_transform(im)
            
    return im_tensor

def predict(image_path, model, cat_to_name, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image_predict= process_image(image_path)
    img = image_predict.unsqueeze(0)
    img = img.to(device)
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        probs = torch.exp(model(img))
        top_probs, top_labels = probs.topk(topk)

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labels = top_labels[0].cpu().numpy()

    top_labels = []
    for label in np_top_labels:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]     
    
    return top_probs, top_labels, top_flowers