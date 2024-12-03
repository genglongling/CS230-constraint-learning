import numpy as np
from mdp import GridMDP
from utils import calculate_kl_divergence, find_unaccrued_features
import pickle
import matplotlib.pyplot as plt
import os, argparse, shutil
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import math

import pandas as pd
import random
import glob
import pandas
import viz
import pyglet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n = 35

# Load tracks, tracksMeta, recordingMeta
tracks_files = glob.glob("inD/*_tracks.csv")
tracksMeta_files = glob.glob("inD/*_tracksMeta.csv")
recordingMeta_files = glob.glob("inD/*_recordingMeta.csv")

# Choose the 00_* files

# Read tracksMeta, recordingsMeta, tracks
tm = pandas.read_csv("./inD/00_tracksMeta.csv").to_dict(orient="records")
rm = pandas.read_csv("./inD/00_recordingMeta.csv").to_dict(orient="records")
t = pandas.read_csv("./inD/00_tracks.csv").groupby(["trackId"], sort=False)

# Normalization
xmin, xmax = np.inf, -np.inf
ymin, ymax = np.inf, -np.inf

bboxes = []
centerpts = []
frames = []
# iterate through groups
for k in range(t.ngroups):

    # Choose the kth track and get lists
    g = t.get_group(k).to_dict(orient="list")

    # Set attributes
    meter_to_px = 1. / rm[0]["orthoPxToMeter"]
    g["xCenterVis"] = np.array(g["xCenter"]) * meter_to_px
    g["yCenterVis"] = -np.array(g["yCenter"]) * meter_to_px
    g["centerVis"] = np.stack([np.array(g["xCenter"]), -np.array(g["yCenter"])], axis=-1) * meter_to_px
    g["widthVis"] = np.array(g["width"]) * meter_to_px
    g["lengthVis"] = np.array(g["length"]) * meter_to_px
    g["headingVis"] = np.array(g["heading"]) * -1
    g["headingVis"][g["headingVis"] < 0] += 360
    g["bboxVis"] = viz.calculate_rotated_bboxes(
        g["xCenterVis"], g["yCenterVis"],
        g["lengthVis"], g["widthVis"],
        np.deg2rad(g["headingVis"])
    )

    # M bounding boxes
    bbox = g["bboxVis"]
    centerpt = g["centerVis"]
    bboxes += [bbox]
    centerpts += [centerpt]
    frames += [g["frame"]]
    xmin, xmax = min(xmin, np.min(bbox[:, :, 0])), max(xmax, np.max(bbox[:, :, 0]))
    ymin, ymax = min(ymin, np.min(bbox[:, :, 1])), max(ymax, np.max(bbox[:, :, 1]))

# normalize
for i in range(len(bboxes)):
    bboxes[i][:, :, 0] = (bboxes[i][:, :, 0]-xmin) / (xmax-xmin) * 1000.
    bboxes[i][:, :, 1] = (bboxes[i][:, :, 1]-ymin) / (ymax-ymin) * 1000.
    centerpts[i][:, 0] = (centerpts[i][:, 0]-xmin) / (xmax-xmin) * 1000.
    centerpts[i][:, 1] = (centerpts[i][:, 1]-ymin) / (ymax-ymin) * 1000.



class DiscreteGrid(viz.Group):
    def __init__(self, x, y, w, h, arr):
        self.arr = arr
        self.itemsarr = np.array([[None for j in range(arr.shape[1])] for i in range(arr.shape[0])])
        self.allpts = [[None for j in range(arr.shape[1])] for i in range(arr.shape[0])]
        self.xsize, self.ysize = w/arr.shape[0], h/arr.shape[1]
        self.colors = {0:(0,0,0,0.5), 1:(1,0,0,0.5), 2:(0,1,0,0.5), 3:(0,0,1,0.5)}
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                pts = [[x+i*self.xsize+self.xsize/10, y+j*self.ysize+self.ysize/10], 
                       [x+(i+1)*self.xsize-self.xsize/10, y+j*self.ysize+self.ysize/10],
                       [x+(i+1)*self.xsize-self.xsize/10, y+(j+1)*self.ysize-self.ysize/10],
                       [x+i*self.xsize+self.xsize/10, y+(j+1)*self.ysize-self.ysize/10]]
                self.allpts[i][j] = pts
                self.itemsarr[i][j] = viz.Rectangle(pts, color = self.colors[arr[i][j]])
        try:
            for pt in constraints["state"]:
                self.itemsarr[pt%n][pt//n].color = (1, 1, 1, 1)
        except:
            pass
        super().__init__(items = self.itemsarr.flatten().tolist())


canvas = viz.Canvas(1000, 1000, id = "000")
canvas.set_visible(False)
pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
arr = np.zeros((n, n))
canvas.items += [DiscreteGrid(20, 60, 1000-30, 1000-60, arr)]

def localize(x, y, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (i, j)
    return (-1, -1)

def delocalize(pt, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if i+j*n == pt:
                return np.array([(x1+x2)/2, (y1+y2)/2])
            
def preprocess_and_filter_trajectories(csv_file, frame_rate=25, x_threshold=80, y_threshold=-80):
    """
    Preprocesses trajectory data from a CSV file into transitions for each vehicle,
    and filters trajectories based on end conditions.

    Parameters:
        csv_file (str): Path to the CSV file containing trajectory data.
        frame_rate (float): Frame rate of the recording in Hz (default: 25 Hz).
        x_threshold (float): Threshold for filtering trajectories based on final x position.
        y_threshold (float): Threshold for filtering trajectories based on final y position.

    Returns:
        filtered_transitions (list): A list of transitions for trajectories that satisfy the conditions.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize the result list
    filtered_transitions = []

    # Process data for each unique trackId
    for track_id, group in df.groupby("trackId"):
        # Sort by frame to ensure correct time sequence
        group = group.sort_values("frame")

        # Check if the last state satisfies the filtering condition
        final_state = group.iloc[-1]
        first_state = group.iloc[0]
        if final_state["xCenter"] < x_threshold and final_state["yCenter"] < y_threshold and (first_state["xCenter"] > 160 or first_state["yCenter"] > -40):
            # Extract relevant columns for processing
            states = group[["xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].values
            #initial_state = [1,0] if states[0][0] < 140 else [0,1]
            # Create transitions (current_state -> next_state)
            for i in range(10, len(states) - 10):
                current_state = states[i:i+10].mean(axis=0)
                next_state = states[i+10:i+20].mean(axis=0)
                #choices = list(range(i - 50, i - 30))
                #j = np.random.choice(choices)
                #while j < 0 or j > len(states) - 10:
                    #j = np.random.choice(choices)
                filtered_transitions.append([np.hstack((current_state, next_state)), 1]) 
                #random_state = states[j:j+10].mean(axis=0)
                #past_state = states[i-10:i].mean(axis=0)
                #filtered_transitions.append([np.hstack((current_state, past_state)), 0])
                if 120 < current_state[0] < 140:
                    filtered_transitions.append([np.hstack((current_state, next_state)), 1]) 
                    filtered_transitions.append([np.hstack((current_state, next_state)), 1]) 
                    filtered_transitions.append([np.hstack((current_state, next_state)), 1]) 


    print(f"Filtered {len(filtered_transitions)} transitions from trajectories that end with x < {x_threshold} and y < {y_threshold}.")
    return filtered_transitions

# Example Usage
input_csv = "./inD/00_tracks.csv"  # Replace with your file path
filtered_transitions = preprocess_and_filter_trajectories(input_csv)

# Example: print the first few filtered transitions
for i, (input_features, label) in enumerate(filtered_transitions[:5]):
    print(f"Filtered Transition {i + 1}: Current {input_features}, Next {label}")

class TransitionPredictionNN(nn.Module):
    def __init__(self, input_dim=14):
        super(TransitionPredictionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Predict 6 values
        )
        self.initialize_weights()  # Call the weight initialization method


    def forward(self, x):
        return self.fc(x)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Apply Xavier initialization for weights
                nn.init.xavier_uniform_(m.weight)
                # Initialize biases to 0
                nn.init.constant_(m.bias, 0)


from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def split_data(transitions, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the transitions into training, validation, and testing sets.

    Parameters:
        transitions (list): List of (current_state, next_state) tuples.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each set.
    """
    # Extract current and next states
    input_states = torch.tensor([np.hstack((t[0][:2])) for t in transitions], dtype=torch.float32)
    #labels = torch.tensor([t[1] for t in transitions], dtype=torch.float32)
    labels = torch.tensor([(t[0][6:8] - t[0][:2]) for t in transitions], dtype=torch.float32)
    #print(input_states[:4], labels[:4])

    # Split the data
    train_x, temp_x, train_y, temp_y = train_test_split(input_states, labels, test_size=(1 - train_ratio))
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=val_ratio / (1 - train_ratio))

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=64, shuffle=False, pin_memory=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=64, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = split_data(filtered_transitions)

def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs):
    """
    Trains the model and evaluates on the validation set.

    Parameters:
        model (nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int): Number of training epochs.

    Returns:
        train_loss_history, val_loss_history: Lists of loss values for training and validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()
            #print(f"Batch {i}: Features shape = {batch_features.shape}, Targets shape = {batch_targets.shape}")
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                #logits = model(batch_features).squeeze(dim=-1)
                #loss = criterion(logits, batch_targets)
                predicted = model(batch_features)  # Shape: (batch_size, 2)
                predicted_norm = torch.nn.functional.normalize(predicted, p=2, dim=1).to(device)
                target_norm = torch.nn.functional.normalize(batch_targets, p=2, dim=1).to(device)
                #print(predicted_norm, target_norm)
                #labels = torch.ones(predicted.size(0)).to(device)  # Move labels to the same device
                # Compute the loss
                #print(predicted, batch_targets)
                loss = criterion(predicted_norm, target_norm)
                #print(loss)
                #loss = criterion(predicted_norm, target_norm, torch.ones(predicted.size(0)))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()
                predicted = model(batch_features)
                predicted_norm = torch.nn.functional.normalize(predicted, p=2, dim=1).to(device)
                target_norm = torch.nn.functional.normalize(batch_targets, p=2, dim=1).to(device)
                #labels = torch.ones(predicted.size(0)).to(device)  # Move labels to the same device

                # Compute the loss
                loss = criterion(predicted_norm, target_norm)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if epoch % 4 == 0:
            trajectories, demons = generate_trajectories(model, count = 100)
            with open('pickles/nn_trajectories_' + str(epoch) + '.pickle', 'wb') as handle:
                pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('pickles/nn_demons_' + str(epoch) + '.pickle', 'wb') as handle:
                pickle.dump(demons, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return train_loss_history, val_loss_history

def test_model(model, test_loader):
    """
    Tests the model on the testing set and returns the average loss.

    Parameters:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for testing data.

    Returns:
        test_loss: Average loss on the testing set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device).float()
            predicted = model(batch_features)
            predicted_norm = torch.nn.functional.normalize(predicted, p=2, dim=1).to(device)
            target_norm = torch.nn.functional.normalize(batch_targets, p=2, dim=1).to(device)
            #labels = torch.ones(predicted.size(0)).to(device)  # Move labels to the same device

            # Compute the loss
            loss = criterion(predicted_norm, target_norm)
            total_loss += loss.item()

    test_loss = total_loss / len(test_loader)
    print(f"Test Loss = {test_loss:.4f}")
    return test_loss

def localize(x, y, grid):
    for i in range(len(grid.allpts)):
        for j in range(len(grid.allpts[0])):
            pt1, pt2, pt3, pt4 = grid.allpts[i][j]
            x1, x2 = pt1[0] - grid.xsize/10, pt2[0] + grid.xsize/10
            y1, y2 = pt2[1] - grid.ysize/10, pt3[1] + grid.ysize/10
            if x1 <= x <= x2 and y1 <= y <= y2:
                return (i, j)
    return (-1, -1)

def getGridPoint(x, y, canvas):
    x = x*meter_to_px
    y = -y*meter_to_px    
    x = (x-xmin) / (xmax-xmin) * 1000.
    y = (y-ymin) / (ymax-ymin) * 1000.
    return localize(x,y,canvas.items[-1])


def getGridPoint2(x, y, grid):
    x = x*meter_to_px
    y = -y*meter_to_px    
    x = (x-xmin) / (xmax-xmin) * 1000.
    y = (y-ymin) / (ymax-ymin) * 1000.   
    i, j = ((x-20)/grid.xsize), ((y-60)/grid.ysize)
    if i < 0 or j < 0 or i > 35 or j > 35:
        return -1, -1
    else:
        return int(i), int(j)

import matplotlib.pyplot as plt

def plot_errors(train_loss, val_loss, test_loss):
    """
    Plots training, validation, and test errors.

    Parameters:
        train_loss (list): Training loss history.
        val_loss (list): Validation loss history.
        test_loss (float): Final test loss.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid()
    plt.show()


def generate_trajectories(model, count = 300):

    for feature, label in random.sample(filtered_transitions, k=count): 
        x, y = feature[0], feature[1]
        paths = []
        demons = []
        trajectories = []
        if paths and x == paths[-1][0][0]:
            continue
        path = [[x, y]]
        trajectories.append([])
        demons.append([])
        for step in range(300):
            dirs = model(torch.tensor([x, y]).to(device).float()).to('cpu')
            norm = dirs[0]*dirs[0] + dirs[1]*dirs[1]
            x = x + dirs[0]/math.sqrt(norm)*0.3
            y = y + dirs[1]/math.sqrt(norm)*0.3
            path.append([x.detach().numpy(), y.detach().numpy()])
            i, j = getGridPoint2(path[-1][0], path[-1][1], canvas.items[-1])
            if i == -1 or j == -1:
                break
            if i == -1 or j == -1:
                break
            if demons[-1] and demons[-1][-1] == i+j*n:
                continue
            demons[-1].append(i+j*n)
            paths.append(path)
            trajectories[-1].append((i, j))
        paths.append(path)
    return trajectories, demons



def train_model_generate_trajs(data, epochs = 50, lr = 0.001):
    
    model = TransitionPredictionNN(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr)   
    # Load and process trajectory data
    train_loader, val_loader, test_loader = split_data(filtered_transitions)
    train_loss, val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs)

    # Initialize model and optimizer

    # Train model
    test_loss = test_model(model, test_loader)


    # Plot loss history
    plot_errors(train_loss[1:], val_loss[1:], test_loss)
    return model



model = train_model_generate_trajs(filtered_transitions, 41, lr = 0.005)


