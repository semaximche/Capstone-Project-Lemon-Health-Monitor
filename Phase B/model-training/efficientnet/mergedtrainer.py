import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import time
import argparse

class Configuration:
    # command line arguments configurations
    input_dir: str
    test_dir: str = None            # Optional
    save_dir: str = None            # Optional
    input_size: int
    batch_size: int
    epochs: int
    
    validate: bool
    pretrained: bool
    
    affine_degrees: int
    affine_translate: float
    
    calc_batch_size: int
    num_workers: int
    learning_rate: float
    momentum: float
    
    # derived configurations
    mean: list[float] = None
    std: list[float] = None
    
    device: torch.device = None
    
    train_size: int = 0
    valid_size: int = 0
    test_size: int = 0
    classes_num: int = 0
    class_names: list[str] = 0
    
    def __init__(self, arguments):
        self.input_dir = arguments.input_dir
        self.test_dir = arguments.test_dir
        self.save_dir = arguments.save_dir
        self.input_size = arguments.input_size
        self.batch_size = arguments.batch_size
        self.epochs = arguments.epochs
        self.validate = arguments.validate
        self.pretrained = arguments.pretrained
        self.affine_degrees = arguments.affine_degrees
        self.affine_translate = arguments.affine_translate
        self.calc_batch_size = arguments.calc_batch_size
        self.num_workers = arguments.num_workers
        self.learning_rate = arguments.learning_rate
        self.momentum = arguments.momentum
        self.mean = arguments.mean
        self.std = arguments.std
        self.complete_configuration()
        
    def complete_configuration(self):
        print(" -- Setting Up Trainer--")
        
        # calculate mean and std if not given in arguments
        if self.mean == None or self.std == None:
            self.mean, self.std = mean_std_images_calc(self.input_dir, self.calc_batch_size)
        
        # get user device
        if torch.cuda.is_available:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # initial load for data
        train_data, valid_data = load_train_data(self.input_dir, self)
        
        # get sizes and numbers
        self.train_size = len(train_data.dataset)
        self.valid_size = len(valid_data.dataset)
        self.full_size = self.train_size + self.valid_size
        self.classes_num = len(train_data.dataset.classes)
        self.class_names = train_data.dataset.classes
        
        # print training information
        print(f"Configuration:\n",
            f"\tData Path:\t\t{self.input_dir}\n",
            f"\tTest Path:\t\t{self.test_dir}\n",
            f"\tDevice:\t\t\t{self.device.type}\n",
            f"\tEpochs Number:\t\t{self.epochs}\n",
            f"\tBatch Size:\t\t{self.batch_size}\n",
            f"\tClasses Number:\t\t{self.classes_num}\n",
            f"\tDataset Mean:\t\t{self.mean}\n",
            f"\tDataset STD:\t\t{self.std}\n")
        
        # generate classes array
        train_balance_array = [0] * self.classes_num
        valid_balance_array = [0] * self.classes_num
        test_balance_array = [0] * self.classes_num
        full_balance_array = [0] * self.classes_num

        for i in range(self.train_size):
            train_balance_array[train_data.dataset.targets[i]] += 1
            full_balance_array[train_data.dataset.targets[i]] += 1
        for i in range(self.valid_size):
            valid_balance_array[valid_data.dataset.targets[i]] += 1
            full_balance_array[valid_data.dataset.targets[i]] += 1
            
        # initial optional load for test, update sizes and arrays
        if self.test_dir != None:
            test_data = load_test_data(self.test_dir, self.batch_size, self.num_workers, self.input_size, self.mean, self.std)
            self.test_size = len(test_data.dataset)
            self.full_size += self.test_size
            
            for i in range(self.test_size):
                test_balance_array[test_data.dataset.targets[i]] += 1
                full_balance_array[test_data.dataset.targets[i]] += 1
            
        # print sizes and balance arrays
        print(f"Dataset:")
        print(f"\tTrain size:\t{self.train_size}\t{train_balance_array}")
        print(f"\tValid size:\t{self.valid_size}\t{valid_balance_array}")
        if self.test_size != 0: print(f"\tTest size:\t{self.test_size}\t{test_balance_array}")
        print(f"\tFull size:\t{self.full_size}\t{full_balance_array}\n")
        print(f"Class names:\n\t{self.class_names}\n")
        
        # display image grids for loaders for sanity check
        # show_dataset_images(train_data)
        # show_dataset_images(valid_data)
        # if self.test_size != 0: show_dataset_images(test_data)

def mean_std_images_calc(images_dir: str, calc_batch_size: int = 64):
    """Calculate mean and standard deviation tensors for an images dataset.
    Works for both single-channel grayscale and multi-channel colour images.
    The output tensors sizes depend on how many channels the images have.

    Args:
        images_dir (str): Path to images directory.
        calc_batch_size (int): Batch size to calculate tensors with. Defaults to 64.

    Returns:
        Tensor,Tensor: mean, std.
    """
    tensor_dataset = datasets.ImageFolder(root=images_dir, transform=transforms.ToTensor())
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=calc_batch_size)
    
    channels_sum, channels_squared_sum, num_batches = 0, 0, len(tensor_dataloader)
    
    # Mean over batch, height and width, but not over the channels.
    print("Calculating mean and std for dataset:")
    for tensor, _ in tqdm(tensor_dataloader, total=num_batches):
        channels_sum += torch.mean(tensor, dim=[0,2,3])
        channels_squared_sum += torch.mean(tensor**2, dim=[0,2,3])
        
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def load_train_data(input_dir: str, config: Configuration):
    """get dataloaders for training and validation sets

    Args:
        input_dir (str): path to directory that has train and val sub-folders
        config (Configuration): configuration object

    Returns:
        DataLoader,DataLoader: training dataloader, validation dataloader
    """
    
    # compose transforms for train/valid dataset
    train_transform = transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.CenterCrop(config.input_size),
        transforms.RandomAffine(degrees=config.affine_degrees, translate=(config.affine_translate, config.affine_translate)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])
    
    test_transform = transforms.Compose([
    transforms.Resize(config.input_size),
    transforms.CenterCrop(config.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
    ])
    
    # open dataset folders and create loaders
    train_dataset = datasets.ImageFolder(root=f"{input_dir}/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=f"{input_dir}/val", transform=test_transform)

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    return train_loader, valid_loader


def load_test_data(test_dir: str, batch_size: int, workers: int, input_size: int, mean: list[float], std: list[float]):
    """get dataloader for testing set

    Args:
        input_dir (str): path to directory that contains testing images
        config (Configuration): configuration object

    Returns:
        DataLoader: testing dataloader
    """
    
    # compose transforms for test dataset
    test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
    
    # open test folder and create loaders
    test_dataset = datasets.ImageFolder(root=f"{test_dir}", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=workers)
    
    return test_loader

def show_dataset_images(dataloader):
    # get one batch from dataloader
    dataloader_iter = iter(dataloader)
    images, labels = next(dataloader_iter)

    img_grid = make_grid(images, normalize=True)
    
    # show images grid with matplotlib with labels in the title
    npimg_grid = img_grid.numpy()
    plt.imshow(np.transpose(npimg_grid, (1, 2, 0)))
    plt.title(', '.join("%d" % labels[i] for i in range(len(images))))
    plt.show()

def get_model(config: Configuration):
    # set up model
    if config.pretrained:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    else:
        model = models.efficientnet_v2_s()
        
    # replace final layer to match classes
    num_features = model.classifier[1].in_features
    model.classifier[1]= nn.Linear(num_features, config.classes_num)
    
    return model

def train_model(config: Configuration, model, criterion, optimzer, scheduler):
    # setting up training
    train_loss = []
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_vacc = 0.0
    best_epoch = 0
    
    print(f" -- Training Model --")
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}:")
        
        train_data, valid_data = load_train_data(config.input_dir, config)
        train_size = len(train_data.dataset)
        valid_size = len(valid_data.dataset)
        
        # training phase
        running_loss = 0.0
        running_corrects = 0
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(train_data), total=len(train_data)):
            # move batch into device and zero optimizer
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimzer.zero_grad()
            
            # compute predictions and loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            
            # backpropagation
            loss.backward()
            optimzer.step()
            
            # gather statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # validataion phase
        if config.validate:
            running_vloss = 0.0
            running_vcorrects = 0
            model.eval()
            with torch.no_grad():
                for i, (vinputs, vlabels) in tqdm(enumerate(valid_data), total=len(valid_data)):
                    # move batch into device and zero optimizer
                    vinputs, vlabels = vinputs.to(config.device), vlabels.to(config.device)
                    
                    # compute predictions and loss
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    _, vpreds = torch.max(voutputs.data, 1)
                    
                    # gather statistics
                    running_vloss += vloss.item() * inputs.size(0)
                    running_vcorrects += torch.sum(vpreds == vlabels.data)
                
        # scheduler step
        before_lr = optimzer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimzer.param_groups[0]["lr"]
        
        # calculate statistics
        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        if config.validate:
            epoch_vloss = running_vloss / valid_size
            epoch_vacc = running_vcorrects.double() / valid_size
        
        # choose best weights whether there is validation or not
        if config.validate:
            if epoch_vacc > best_vacc:
                best_epoch = epoch
                best_vacc = epoch_vacc
                best_model_wts = model.state_dict()
            if epoch_vacc > 0.999:
                break
        else:
            if epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
            if epoch_acc > 0.999:
                break
            
        # save model each epoch
        model_out_path = config.save_dir + '/efficientnet_best_epoch_' + str(best_epoch) + '.pth'
        torch.save(model, model_out_path)
        print(f'Saved model at {model_out_path}\n')
    
        # print results
        print('Train Loss: {:.4f} Train Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
        if config.validate:
            print('Valid Loss: {:.4f} Valid Accuracy: {:.4f}'.format(epoch_vloss, epoch_vacc))
    
    # after training
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return best_model_wts

def test_model(model_path, test_dir, class_names, batch_size: int, workers: int, input_size: int, mean: list[float], std: list[float]):
    print(f" -- Testing Model --")
    
    # get device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # load model and test dataset
    model = torch.load(model_path, weights_only=False)
    test_data = load_test_data(test_dir, batch_size, workers, input_size, mean, std)
    test_size = len(test_data.dataset)
    
    # test Model
    running_corrects = 0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(test_data), total=len(test_data)):
            # move batch into device and zero optimizer
            inputs, labels = inputs.to(device), labels.to(device)
                    
            # compute predictions and loss
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
                    
            # gather statistics
            running_corrects += torch.sum(preds == labels.data)
            
    # calculate statistics
    test_acc = running_corrects.double() / test_size
    print('Test Accuracy: {:.4f}'.format(test_acc))
    
def parse_args():
    """Creates the Argument Parser for the trainer and returns namespace containing all input arguments.

    Returns:
        Namespace: Namespace of all the given arguments.
    """
    parser = argparse.ArgumentParser(prog='EfficientNet Trainer',
                                     description='Automatically trains EfficientNet Models on custom image datasets',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('input_dir', type=str, default=None, help='input folder path, should contain train, val subfolders')
    
    parser.add_argument('-t', '--test-dir', type=str, default=None, help='test set folder path, should contain test images')
    parser.add_argument('-o', '--save-dir', type=str, default='.', help='folder to save model to. would be called efficientnet.pth after running')
    parser.add_argument('-s', '--input_size', type=int, default=224, help='image size for model input, input will be resized to this')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='model input batch size, higher number requires more device memory')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs to train the model for')
    
    parser.add_argument('--pretrained', action='store_true', help="use ImageNet1kv1 weights")
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help="dont use any pretraine weights")
    parser.set_defaults(pretrained=True)
    
    parser.add_argument('--validate', action='store_true', help="validate after each training epoch")
    parser.add_argument('--no-validate', dest='validate', action='store_false', help="dont validate after each training epoch")
    parser.set_defaults(validate=False)
    
    parser.add_argument('--affine_degrees', type=int, default=0, help='maximum +- degrees for random rotation transform, between 0 and 180')
    parser.add_argument('--affine_translate', type=float, default=0.05, help='maximum random horizontal/vertical random translation')
    
    parser.add_argument('--calc-batch-size', type=int, default=64, help='batch size for dataset mean and std calculation')
    parser.add_argument('--num-workers', type=int, default=3, help='workers number for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='initial learning rate to start training on')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum to use for training')
    
    parser.add_argument('--mean', type=float, nargs='*', help='mean values for the dataset as a list')
    parser.add_argument('--std', type=float, nargs='*', help='standard deviation values for the dataset as a list')

    return parser.parse_args()


if __name__ == '__main__':
    config = Configuration(parse_args())
    
    # get model, criterion, optimizer and scheduler
    model = get_model(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD((model.parameters()), lr=config.learning_rate, momentum=config.momentum, weight_decay=0.0004)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9, last_epoch=-1)
    
    # move model and criterion to user device
    model.to(device=config.device)
    criterion.to(device=config.device)
    
    # train model
    best_weights = train_model(config, model, criterion, optimizer, scheduler)
    
    # save best model
    model.load_state_dict(best_weights)
    model_out_path = config.save_dir + '/best_efficientnet.pth'
    torch.save(model, model_out_path)
    print(f'Saved model at {model_out_path}\n')
    
    # test model
    test_model(model_out_path, config.test_dir, config.class_names, config.batch_size, config.num_workers, config.input_size, config.mean, config.std)