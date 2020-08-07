import torch
import torchvision.models as models
import time
import copy
from collections import defaultdict
import pandas as pd
import numpy as np
import image_util as kmg
import dataloader
# Python's native libraries
import time
import os
import copy
from collections import defaultdict

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import cv2 as cv  # OpenCV

# numeric and plotting libraries
import numpy as np
import matplotlib.pyplot as plt

def load_data():

     #Joining together the needed paths
    path_vid =r"D:/20bn-jester-v2"
    BASE_PATH = "D:"
    #path_model = os.path.join(data_root, data_model, model_name)
    path_labels = BASE_PATH + '/jester-v1-labels.csv'
    path_train =  BASE_PATH + '/jester-v1-train.csv'
    path_test =  BASE_PATH + '/jester-v1-test.csv'
    path_val = BASE_PATH + '/jester-v1-validation.csv'

    inp_shape   = (16,) + (3,) + (112,112)

    data = dataloader.DataLoader(path_vid, path_labels, path_val, path_val)
    target_size =  (112,112)
    batch_size = 32
    nb_frames = 16
    skip = 1

    #Creating the generators for the training and validation set
    gen = kmg.ImageDataGenerator()
    gen_train = gen.flow_video_from_dataframe(data.train_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
    gen_val = gen.flow_video_from_dataframe(data.val_df, path_vid, path_classes=path_labels, x_col='video_id', y_col="label", target_size=target_size, batch_size=batch_size, nb_frames=nb_frames, skip=skip, has_ext=True)
        
    resnet101 = models.resnet101(pretrained=True)

    finetuned_model = copy.deepcopy(resnet101)

    resnet101.fc = torch.nn.Linear(resnet101.fc.in_features, 27)

    opti = torch.optim.SGD(resnet101.parameters(),lr=0.01, momentum=0.9, nesterov=False)
    # resnet101.compile(optimizer=opti,
    #                 loss="categorical_crossentropy",
    #                 metrics=["accuracy"]) 
    nb_sample_train = data.train_df["video_id"].size
    nb_sample_val   = data.val_df["video_id"].size

    # WORKSHOP CODE
    torch.cuda.empty_cache()
    num_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet101
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, finetuned_model.parameters()))
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    metrics = defaultdict(list)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            print(gen_train)

            # Iterate over data.
            for inputs, labels in gen_train:

                inputs = torch.flatten(torch.from_numpy(inputs),start_dim=1, end_dim=2).transpose(1, 3)
                labels = torch.from_numpy(labels)
                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss /nb_sample_train
            epoch_acc = float(running_corrects) / nb_sample_train
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            metrics[phase+"_loss"].append(epoch_loss)
            metrics[phase+"_acc"].append(epoch_acc)
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
    print('Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


if __name__ == '__main__':
    torch.hub.set_dir("D:")
    resnet18 = models.resnet18(pretrained=True)
    resnet101 = models.resnet101(pretrained=True)

    resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, 2)
    load_data()

