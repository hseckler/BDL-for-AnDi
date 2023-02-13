#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import andi_code as andi
import csv


# In[2]:


#convert trajectory to increments and normalize their standarddeviation to 1
def normalize(trajectory):
    #print(trajectory)
    
    mean = np.mean(trajectory)
    std = np.std(trajectory)
    
    
    return (trajectory-mean)/std


def convert_to_increments(trajectory):
    delta = trajectory[1:]-trajectory[:-1]
    d_std = np.std(delta)
    
    if d_std >= 1e-5:
        delta /= d_std
    
    #mean = np.mean(trajectory)
    #delta -= mean
    
    return delta

class AnDi_Regression_dataset(Dataset):

    def __init__(self, path = "datasets/1dim_100lenght/", maxsamples = None, skiprows = 0):
        #load input trajectories and alpha values of "task1"
        trajectories = np.loadtxt(path+"task1.txt", delimiter=";", 
                                  dtype = np.float32, max_rows = maxsamples, skiprows = skiprows)
        alpha_values = np.loadtxt(path+"ref1.txt", delimiter=";", 
                                  dtype = np.float32, max_rows = maxsamples, skiprows = skiprows)
            
        #configer output targets
        alpha_values = torch.from_numpy(alpha_values)
        self.targets = alpha_values[:,1:] #first value only tells dimension, not of interest
        
        
        #number of samples/trajectories
        self.n_samples = alpha_values.shape[0]
        
        #remove dimension entry from list (first entry in each trajectory)
        dim = int(trajectories[0,0])
        if dim > 1:
            raise("This dataset is not yet set up for higher dim!")
        
        trajectories = trajectories[:,1:].reshape((self.n_samples,-1,dim))
        #normalize the trajectories; revisit this if dimension ever gets larger then 1!!
        trajectories = np.apply_along_axis(normalize, 1, trajectories)
        
        
        #configer input trajectories
        trajectories = torch.from_numpy(trajectories)
        #reshape
        self.trajectories = trajectories.view((self.n_samples,-1,dim)) 
        
            
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.trajectories[index], self.targets[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class AnDi_Classification_dataset(Dataset):

    def __init__(self, path = "datasets/1dim_100lenght/", maxsamples = None, skiprows = 0, no_targets = False):
        #load input trajectories and alpha values of "task2"
        trajectories = np.loadtxt(path+"task2.txt", delimiter=";",
                                  dtype = np.float32, max_rows = maxsamples, skiprows = skiprows)
        
        labels = np.loadtxt(path+"ref2.txt", delimiter=";",
                                dtype = int, max_rows = maxsamples, skiprows = skiprows)
            
        #configer output targets
        labels = torch.from_numpy(labels).long()
        self.targets = labels[:,1:] #first value only tells dimension, not of interest
         
        #number of samples/trajectories
        self.n_samples = labels.shape[0]
        
        #remove dimension entry from list (first entry in each trajectory)
        dim = int(trajectories[0,0])
        if dim > 1:
            raise("This dataset is not yet set up for higher dim!")
        
        trajectories = trajectories[:,1:].reshape((self.n_samples,-1,dim))
        #normalize the trajectories
        trajectories = np.apply_along_axis(normalize, 1, trajectories)
        
        
        #configer input trajectories
        trajectories = torch.from_numpy(trajectories)
        #first value tells dimension of trajectory
        self.trajectories = trajectories.view((self.n_samples,-1,dim)) 
        
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.trajectories[index], self.targets[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

    
#load only trajectory data and also allow for trajectories of different lenght in dataset
class AnDi_trajectories_only(Dataset):
    
    def __init__(self, path = "datasets/andi_challenge_dataset/", name = "task1.txt", dim = 1):
        #open file
        savename = path+name
        t = csv.reader(open(savename,"r"),delimiter=";", 
                                  lineterminator = "\n", quoting = csv.QUOTE_NONNUMERIC)
        
        #load trajectories from index start to end
        self.trajectories = []
        for idx, (trajs) in enumerate(zip(t)):
            if trajs[0][0] == dim:
                trajectory = np.asarray(trajs[0][1:], dtype = np.float32)
                self.trajectories.append(torch.from_numpy(normalize(trajectory)))
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.trajectories[index]
    
    # we can call len(dataset) to return the size
    def __len__(self):
        return len(self.trajectories)
    
#loading a super dataset containing trajectories and labels for models, exponents and noise
class AnDi_super_dataset(Dataset):
    def __init__(self, path = "datasets/super/1dim_100lenght/", fname = "andiset10000.txt", dim=1):
        if dim != 1:
            raise("super dataset not yet set up for higher dimensions!")
        
        #load
        dataset = np.loadtxt(path+fname, dtype = np.float32)
        #split data
        model_labels = dataset[:,0].astype(int)
        alpha_values = dataset[:,1]
        noise_values = dataset[:,2]
        trajectories = dataset[:,3:]
        
        #number of samples is len of any of the above
        self.n_samples = len(alpha_values)
        
        #normalize trajectories
        trajectories = np.apply_along_axis(normalize, 1, trajectories)
        trajectories = trajectories.reshape((self.n_samples,-1,dim))
        
        #to torch and set as self attributes
        self.model_labels = torch.from_numpy(model_labels).long()
        self.exponent_values = torch.from_numpy(alpha_values).view(-1,1)
        self.noise_values = torch.from_numpy(noise_values)
        self.trajectories = torch.from_numpy(trajectories)
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.model_labels[index],self.exponent_values[index],self.noise_values[index],self.trajectories[index]
    
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def to(self, device):
        self.model_labels = self.model_labels.to(device)
        self.exponent_values = self.exponent_values.to(device)
        self.noise_values = self.noise_values.to(device)
        self.trajectories = self.trajectories.to(device)
        
#create a standard andi dataset (classification or regression) from saved trajectories        
class AnDi_dataset_from_saved_trajs(Dataset):
	def __init__(self, path = "datasets/trajectories/", task = 1, dim = 1, N_total = 100000, T = 100, N_save = 10000, use_increments = False):
		if type(T) == type(100):
			min_T = T
			max_T = T+1
		else: #allow for T input as tuple, beeing min and max trajectory length
			min_T = T[0]
			max_T = T[1]
		#use andi tool to create dataset from saved trajectories
		AD = andi.andi_datasets()
		X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save)
		if task == 1:
			trajectories = X1[dim-1]
			labels = Y1[dim-1]
		elif task == 2:
			trajectories = X2[dim-1]
			labels = Y2[dim-1]
		
		self.n_samples = len(labels)
		trajectories = np.asarray(trajectories).reshape((self.n_samples,dim,T)).transpose(0,2,1)
		#normalize trajectories
		if type(T) == type(100):
			trajectories = np.apply_along_axis(normalize, 1, trajectories)
			if use_increments == True:
				trajectories = np.apply_along_axis(convert_to_increments, 1, trajectories)
			trajectories = trajectories.reshape((self.n_samples,-1,dim))
		else:
			for i in range(len(trajectories)):
				trajectories[i] = np.asarray(normalize(trajectories[i]))
				if use_increments == True:
					trajectories[i] = convert_to_increments(trajectories[i])
			
		#to torch and set as self attributes
		if task == 2:
			self.targets = torch.from_numpy(np.asarray(labels)).long()
		elif task == 1:
			self.targets = torch.from_numpy(np.asarray(labels)).view(-1,1).float()
		if type(T) == type(100):
			self.trajectories = torch.from_numpy(trajectories).float()
		else:
			for i in range(len(trajectories)):
				trajectories[i] = torch.from_numpy(trajectories[i]).float()
				self.trajectories = trajectories
	
	# support indexing such that dataset[i] can be used to get i-th sample
	def __getitem__(self, index):
		return self.trajectories[index], self.targets[index]
		
	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_samples



#loading a super dataset containing trajectories and labels for models, exponents and noise
#this is the version for doing so from saved trajectories
class AnDi_super_dataset_from_saved_trajs(Dataset):
	def __init__(self, path = "datasets/trajectories/testset/", task = 1, dim = 1, N_total = 10000, T = 100, N_save = 500, use_increments = False):
		if type(T) == type(100): #generate at just one trajectory length
			min_T = T
			max_T = T+1
		else: #allow for T input as tuple, beeing min and max trajectory length
			min_T = T[0]
			max_T = T[1]
		#use andi tool to create dataset from saved trajectories
		AD = andi.andi_datasets()
		X1, Y1, X2, Y2, X3, Y3 = AD.andi_dataset_superlabeled(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save)
		
		if task == 1:
			trajectories = X1[dim-1]
			labels = np.asarray(Y1[dim-1])
		elif task == 2:
			trajectories = X2[dim-1]
			labels = np.asarray(Y2[dim-1])
		
		self.n_samples = len(labels)
		trajectories = np.asarray(trajectories).reshape((self.n_samples,dim,T)).transpose(0,2,1)
		
		#normalize trajectories
		if type(T) == type(100):
			trajectories = np.apply_along_axis(normalize, 1, trajectories)
			if use_increments == True:
				trajectories = np.apply_along_axis(convert_to_increments, 1, trajectories)
			trajectories = trajectories.reshape((self.n_samples,-1,dim))
		else:
			for i in range(len(trajectories)):
				trajectories[i] = np.asarray(normalize(trajectories[i]))
				if use_increments == True:
					trajectories[i] = convert_to_increments(trajectories[i])
		
		#to torch and set as self attribute
		self.models = torch.from_numpy(labels[:,0]).long()
		self.exponent_values = torch.from_numpy(labels[:,1]).view(-1,1).float()
		self.noise_values = torch.from_numpy(labels[:,2]).float()
		
		if type(T) == type(100):
			self.trajectories = torch.from_numpy(trajectories).float()
		else:
			for i in range(len(trajectories)):
				trajectories[i] = torch.from_numpy(trajectories[i]).float()
				self.trajectories = trajectories
		
	# support indexing such that dataset[i] can be used to get i-th sample
	def __getitem__(self, index):
		return self.models[index], self.exponent_values[index], self.noise_values[index], self.trajectories[index]
		
	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_samples
    
class SingleModel_dataset_from_saved_trajs(Dataset):
	def __init__(self, path = "datasets/trajectories/", task = 1, dim = 1, N_total = 100000, T = 100, N_save = 10000, use_increments = False, model = 0):
		task = 1 #since there is only one model classification is not needed, only task 1 therefore
		
		if type(T) == type(100):
			min_T = T
			max_T = T+1
		else: #allow for T input as tuple, beeing min and max trajectory length
			min_T = T[0]
			max_T = T[1]
		           
		#use andi tool to create dataset from saved trajectories
		AD = andi.andi_datasets()
		if type(model) == type(1):   
			X1, Y1 = AD.andi_dataset_onemodel(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save, model = model, superset = False)
			Xs1, Ys1 = X1[dim-1],Y1[dim-1]
		else:
			Xs1, Ys1 = [],[]
			for i in range(len(model)):
				#to maintain a balanced data with respect to the anomalous exponent 
				#only choose e.g. half as many attm trajs as sbm
				if model == 0 or model == 1 or model == 3: #attm and ctrw and lw
					n_exp = 20/40
				elif model == 2: #fbm not ballistic!
					n_exp = 39/40
				else: #sbm
					n_exp = 40
				X1, Y1 = AD.andi_dataset_onemodel(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save[model[i]], model = model[i], superset = False)
				try: 
					Xs1 = np.concatenate((Xs1,X1[dim-1][0:int(n_exp*N_total-0.5)]))
				except:
					Xs1 = X1[dim-1][0:int(n_exp*N_total-0.5)]
				try:
					Ys1 = np.concatenate((Ys1,Y1[dim-1][0:int(n_exp*N_total-0.5)]))
				except:
					Ys1 = Y1[dim-1][0:int(n_exp*N_total-0.5)]
		trajectories = Xs1
		labels = np.asarray(Ys1)
		
		self.n_samples = len(labels)
		trajectories = np.asarray(trajectories).reshape((self.n_samples,dim,T)).transpose(0,2,1)
		#normalize trajectories
		if type(T) == type(100):
			trajectories = np.apply_along_axis(normalize, 1, trajectories)
			if use_increments == True:
				trajectories = np.apply_along_axis(convert_to_increments, 1, trajectories)
			trajectories = trajectories.reshape((self.n_samples,-1,dim))
		else:
			for i in range(len(trajectories)):
				trajectories[i] = np.asarray(normalize(trajectories[i]))
				if use_increments == True:
					trajectories[i] = convert_to_increments(trajectories[i])
			
		#to torch and set as self attributes
		self.targets = torch.from_numpy(np.asarray(labels)).view(-1,1).float()
		if type(T) == type(100):
			self.trajectories = torch.from_numpy(trajectories).float()
		else:
			for i in range(len(trajectories)):
				trajectories[i] = torch.from_numpy(trajectories[i]).float()
				self.trajectories = trajectories
	
	# support indexing such that dataset[i] can be used to get i-th sample
	def __getitem__(self, index):
		return self.trajectories[index], self.targets[index]
		
	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_samples
    
class SingleModel_superdataset_from_saved_trajs(Dataset):
	def __init__(self, path = "datasets/trajectories/", task = 1, dim = 1, N_total = 100000, T = 100, N_save = 10000, use_increments = False, model = 0):
		task = 1 #since there is only one model classification is not needed, only task 1 therefore
		
		if type(T) == type(100):
			min_T = T
			max_T = T+1
		else: #allow for T input as tuple, beeing min and max trajectory length
			min_T = T[0]
			max_T = T[1]
		           
		#use andi tool to create dataset from saved trajectories
		AD = andi.andi_datasets()
		if type(model) == type(1):
			X1, Y1 = AD.andi_dataset_onemodel(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save, model = model, superset = True)
			Xs1, Ys1 = X1[dim-1],Y1[dim-1]
		else:
			Xs1, Ys1 = [],[]
			for i in range(len(model)):
				#to maintain a balanced data with respect to the anomalous exponent 
				#only choose e.g. half as many attm trajs as sbm
				if model == 0 or model == 1 or model == 3: #attm and ctrw and lw
					n_exp = 20/40
				elif model == 2: #fbm not ballistic!
					n_exp = 39/40
				else: #sbm
					n_exp = 40      
				    
				X1, Y1 = AD.andi_dataset_onemodel(N = N_total, tasks = [task], dimensions = [dim], min_T = min_T, max_T = max_T, load_trajectories = True, path_trajectories = path, N_save = N_save[model[i]], model = model[i], superset = True)
				try: 
					Xs1 = np.concatenate((Xs1,X1[dim-1][0:int(n_exp*N_total-0.5)]))
				except:
					Xs1 = X1[dim-1][0:int(n_exp*N_total-0.5)]
				try:
					Ys1 = np.concatenate((Ys1,Y1[dim-1][0:int(n_exp*N_total-0.5)]))
				except:
					Ys1 = Y1[dim-1][0:int(n_exp*N_total-0.5)]
		trajectories = Xs1
		labels = np.asarray(Ys1)
		#print(labels)
		self.n_samples = len(labels)
		trajectories = np.asarray(trajectories).reshape((self.n_samples,dim,T)).transpose(0,2,1)
		#normalize trajectories
		if type(T) == type(100):
			trajectories = np.apply_along_axis(normalize, 1, trajectories)
			if use_increments == True:
				trajectories = np.apply_along_axis(convert_to_increments, 1, trajectories)
			trajectories = trajectories.reshape((self.n_samples,-1,dim))
		else:
			for i in range(len(trajectories)):
				trajectories[i] = np.asarray(normalize(trajectories[i]))
				if use_increments == True:
					trajectories[i] = convert_to_increments(trajectories[i])
			
		#to torch and set as self attributes
		self.models = torch.from_numpy(labels[:,0]).long()
		self.exponent_values = torch.from_numpy(labels[:,1]).view(-1,1).float()
		self.noise_values = torch.from_numpy(labels[:,2]).float()
        
		if type(T) == type(100):
			self.trajectories = torch.from_numpy(trajectories).float()
		else:
			for i in range(len(trajectories)):
				trajectories[i] = torch.from_numpy(trajectories[i]).float()
				self.trajectories = trajectories
	
	# support indexing such that dataset[i] can be used to get i-th sample
	def __getitem__(self, index):
		return self.models[index], self.exponent_values[index], self.noise_values[index], self.trajectories[index]
		
	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_samples