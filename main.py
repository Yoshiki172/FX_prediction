import os
import shutil
import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from sklearn import model_selection
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import timm
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
from datetime import datetime, timedelta
from datetime import datetime
from tqdm import tqdm
import numpy as np
parser = argparse.ArgumentParser(description='FX prediction program using pytorch')
parser.add_argument('--train', action='store_true')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Make_Dataset(Dataset):
    def __init__(self, image, label_list,date_list, phase=None):

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256)
        ])
        self.image = image
        self.label_list = torch.tensor(label_list,dtype=torch.long)
        self.phase = phase
        self.date_list = date_list

    def __getitem__(self,index):
        # Load and preprocess the index-th image
        image = self.image[index]
        # Get the index-th label
        label = self.label_list[index]
        date = self.date_list[index]
        return image, label, date

    def __len__(self):
        return len(self.image)

def print_ext(updown, rate, data):
    UPtarget = 1 if updown == 'UP' else 0

    # Extract rows from the data frame that match the condition (Pred_zero or Pred_one exceeds the threshold)
    filtered_data = data[((data['Pred_down'] > rate) | (data['Pred_up'] > rate)) & (data['Label'] == UPtarget)]

    if UPtarget == 1:
      target = filtered_data['Pred_up']
    else:
      target = filtered_data['Pred_down']
    # Number of rows matching the condition that were correctly predicted
    correct_predictions = filtered_data[(target > rate)]

    counter = len(correct_predictions)

    total_counter = len(filtered_data)
    print(filtered_data)
    # Display the result
    if total_counter > 0:
        accuracy = counter / total_counter
        print('Accuracy: {:.4f} %'.format(accuracy * 100))
    else:
        print('Accuracy: N/A - division by zero')
    print('Number of correct answers within the predicted: {}'.format(counter))
    print('Number of {} exceeding the threshold {}: {}'.format(updown,rate, total_counter))
    print("--------------------------------------------------------------")

def generate_image(data: pd.DataFrame, folder_path: str = './predict_target') -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The directory '{folder_path}' has been created!")
    else:
        print(f"The directory '{folder_path}' has already exist!")
    
    data['ImageName'] =  [folder_path +'/'+ str(data.index[i].year)+ '-'  + str(data.index[i].month) + '-' + str(data.index[i].day) + '.png' for i in range(len(data))]
    
    short_window = 5   # 5-day moving average
    long_window = 20   # 20-day moving average
    data['SMA5'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['SMA20'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    # Chart Style Setting
    mc = mpf.make_marketcolors(up='#77d879', down='#db3f3f', inherit=True)
    s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc, gridstyle='')

    figsize = 2.56, 2.56
    # Generate Charts
    seq_len = 40

    for i in tqdm(range(len(data) - seq_len + 1)):
        c = data.iloc[i:i + seq_len]

        # Additional plots of moving average line
        add_plot = [
            mpf.make_addplot(c['SMA5'], color='blue', alpha=0.5),
            mpf.make_addplot(c['SMA20'], color='red', alpha=0.5),
        ]
        
        fig, ax = mpf.plot(c, type='candle', style=s, addplot=add_plot, figsize=figsize, returnfig=True)

        # Axis setting
        ax[0].set_xticklabels([])
        ax[0].set_yticklabels([])
        ax[0].xaxis.set_visible(False)
        ax[0].yaxis.set_visible(False)

        # Save file as an image
        pngfile = data['ImageName'][i + seq_len - 1]
        fig.savefig(pngfile, pad_inches=0.1, transparent=False,bbox_inches='tight')
        plt.close(fig)

    print("Candlestick charts and moving averages have been successfully generated!")
    return data

    
# "future_num" days after the close predicted up or down
def prepare(day: int = 60,future_num: int = 1) -> None:
    # Get the current date and format the date into a string in 'yyyy-mm-dd' format
    current_date = datetime.now()
    start_date = current_date - timedelta(days=day)
    current_date_str = current_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    data = yf.download('EURJPY=X', start=start_date_str, end=current_date_str)
    data['Up'] = 0
    
    for i in range(len(data)-future_num):
        if data['Close'][i] < data['Close'][i+future_num]:
            data['Up'][i] = 1
    
    data = generate_image(data)
    
    x_val = data['ImageName'].values[:]
    y_val = data['Up'].values[:]
    z_val = np.array(data.index[:].strftime('%Y-%m-%d').tolist())
    
    X_val = []
    Y_val = []
    Z_val = []
    transform = transforms.Compose([transforms.ToTensor()])
    for image_file,label,date in zip(x_val,y_val,z_val):
        try:
            image = Image.open(image_file)
            image = image.convert('RGB')
            image = transform(image)
            X_val.append(image)
            Y_val.append(label)
            Z_val.append(date)
        except FileNotFoundError as e:
            print(e)
            
    val_dataset = Make_Dataset(X_val,Y_val,Z_val,phase='val')
    
    return val_dataset,data
    

def validation(val_dataset, data):
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    pred_list = np.empty([1, 2])

    i = 0
    input_list = []
    output_list = []
    sample_list = []
    for inputs, labels,dates in val_dataloader:
        i += 1
        #for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            sm = nn.Softmax(dim=1)
            outputs2 = np.array(sm(outputs.cpu()))
            pred_list = np.vstack((pred_list,outputs2))
            output_list.append(outputs2)
            input_list.append(labels)
            labels_np = labels.cpu().numpy()  # Convert tensor to NumPy array
            sample_list.append([labels_np,outputs2,dates])
    data = {'Label': [], 'Pred_down': [], 'Pred_up': [], 'Date': []}

    # Analyze each element of sample_list and add it to the data list
    for item in sample_list:
        data['Label'].append(item[0][0])
        data['Pred_down'].append(item[1][0][0])
        data['Pred_up'].append(item[1][0][1])
        data['Date'].append(item[2][0])

    # Converted to Pandas data frame
    data = pd.DataFrame(data)
    return data

if __name__ == "__main__":
    args = parser.parse_args()
    
    model = models.efficientnet_v2_s(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('base_model/model.pth'))
    model = model.to(device)
    if args.train:
            print('Before Mounting')
    else:
        model.eval()
    val_dataset,data = prepare(day=60)
    data = validation(val_dataset,data)
    print_ext('UP', 0.5, data)
    print_ext('DOWN', 0.5, data)
    print_ext('UP', 0.7, data)
    print_ext('DOWN', 0.7, data)
    print_ext('UP', 0.8, data)
    print_ext('DOWN', 0.8, data)
    print_ext('UP', 0.9, data)
    print_ext('DOWN', 0.9, data)
    shutil.rmtree('predict_target')
