#2.2 Data Preprocessing

# imports
import os
import pandas as pd
import torch

#2.2.1 Reading the Dataset

# NOTE: data_file already created previously
data = pd.read_csv(os.path.join("..", 'data', 'house_tiny.csv')) # reads a csv(comma seperated values) so you can manipulate it
print(data)


#2.2.2 Data Preperation

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2] # iloc is the indec based location, similar to how the data manipulation in the Data Manipulation file
inputs = pd.get_dummies(inputs, dummy_na=True) # this replaces the single column of NaN and Slate to two columns with bool to see if one is nan or slate
print(inputs)

inputs = inputs.fillna(inputs.mean()) # this fills the numRooms NA with the average of the numRooms
print(inputs)

#2.2.3 Conversion to the Tensor Format

X = torch.tensor(inputs.to_numpy(dtype=float)) # this shows how to convert pandas into numpy and then tensor EXTREMELY USEFUL
y = torch.tensor(targets.to_numpy(dtype=float)) # this shows hot to convert pandas into numpy which then gets converted into tensor
print(X, y)

# 2.2.5 Exercises

# Excercise 1
path = '/Users/fm/Downloads/abalone.data'
#from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
#abalone = fetch_ucirepo(id=1) 
  
# data (as pandas dataframes) 
#X = abalone.data.features 
#y = abalone.data.targets 
  
# metadata 
#print(abalone.metadata) 
  
# variable information 
#print(abalone.variables)

#ERRORS HERE - HELP HERE, loading data using os and pandas .data, .name file?

# Excercise 2
# Going to use existing tiny_house.csv file for this and not big dataset

numRooms_in_data = data["NumRooms"]
prices_in_data = data["Price"]
roofType = data["RoofType"]
print(numRooms_in_data, prices_in_data, roofType)

#Excercise 3
# I would suppose you could load a large dataset with this method. Computer memoty might be an issue.
# If dataset too big it might take too long to process the data and then manipulate it.




