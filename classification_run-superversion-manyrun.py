#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.optim import lr_scheduler
from swag.posteriors import swag as swag

from load_andi_dataset import *
from LSTM_Neural_Network import *
from swag_lr_scheduler import *


#Ts = [10,25,50,100,250,500,999]#[10,25,50,100,250,500,999]
Ts = [999]

dim = int(input("Which dimension? Options are 1,2,3: "))
if dim not in [1,2,3]:
    dim = 1
print("Using data of dimension " + str(dim))

for T in Ts:
    print(f"Training models for T={T} in {dim}d...")
    print("Loading data...")

    #setup data using super dataset
    #T = 100
    noise_T = T
    N_train = int(1e6)
    if T == 999:
        N_train = int(5e5)
    N_test = 10000
    use_increments = True


    #loading from saved trajectories, allows for only one dataset of trajectories usable for all trajectory lenghts
    if dim == 1:
        train_path = "datasets/trajectories/"
    else:
        train_path = f"datasets/trajectories/{dim}d/"
    train_dataset = AnDi_dataset_from_saved_trajs(path = train_path, task = 2, dim = dim, N_total = N_train,
                                                  T = T, N_save = [16000,16000,10000,16000,10000], 
                                                  use_increments = use_increments)
    test_path = train_path+"testset/"
    test_dataset = AnDi_dataset_from_saved_trajs(path = test_path, task = 2, dim = dim, N_total = N_test, 
                                                 T = T, N_save = 500, use_increments = use_increments)


    print(len(train_dataset),len(test_dataset))
    print("Finished loading data")





    for manyrunning in range(25):

        #randomly choose hyperparameters i want to observe
        swag_lr = 1e-4#np.random.choice([1e-4,2e-4,5e-5])#2e-4#np.random.choice([0.06,0.03,0.01,0.005])
        cyclic = True

        # Device configuration, run on gpu if available
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Hyper-parameters 
        input_dim = dim # 1D input sequence
        LSTM_size = [128,128,64]#64
        hidden_size = 20
        output_dim = 5 #output size
        num_epochs = 65
        batch_size = 512

        #optimizer hyper-paras
        lr_init = 2e-3#initial learning rate (max learning rate)
        momentum = 0.9 #contribution of earlier gradient to next gradient
        weight_decay = 1e-4 #contribution L2 norm of weights to loss
        cyclic_multiplier = 4 #multiplier for the time it takes for one half of a cyclic period: e.g. 2 episodes


        #parameter choices for swag
        swag_start = 55 #when to start swag epoches (needs to be smaller than num epochs)
        swag_update_freq = 10 #number of times swag estimate is updated per epoch(x updates per epoch)
        if T>=500:
            swag_start = 65
            num_epochs = 75
            swag_update_freq = 10
        if T == 999:
            swag_start = 150
            num_epochs = 170
            swag_update_freq = 5
            cyclic_multiplier = 8
        #swag_lr = 0.01 #swag learning rate


        # In[3]:





        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, pin_memory=False, num_workers = 3)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=False, pin_memory=False, num_workers = 3)




        # In[5]:


        ############## TENSORBOARD ########################
        #to view the tensorboard you need to run "tensorboard --logdir=runs" in console 
        #while in the directory of this file and open the link!
        import sys
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter
        import os

        #specify path for tensorboard
        i = 0 
        while os.path.exists(f"runs/classification/{dim}d/{T}lenght/final_andiclassi_cyclicdecay_{swag_start}-{num_epochs}_{lr_init:.4f}-{swag_lr:.5f}_mom{momentum:.2f}_batch{batch_size}swag%s" % i):
            i += 1
        tensorboardpath = f"runs/classification/{dim}d/{T}lenght/final_andiclassi_cyclicdecay_{swag_start}-{num_epochs}_{lr_init:.4f}-{swag_lr:.5f}_mom{momentum:.2f}_batch{batch_size}swag%s" % i
        #overwritten path (for custom name)
        #tensorboardpath = "runs/classification/andiclassi_swagrun_25-35_lr0.15-0.15"
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(tensorboardpath)
        ###################################################


        # In[ ]:


        #lr_init = 0.3

        #define model
        model = LSTM_Classification(input_dim, output_dim, hidden_size = hidden_size, LSTM_size=LSTM_size).to(device)

        #define swag model
        swag_model = swag.SWAG(LSTM_Classification, subspace_type = 'covariance', 
                               subspace_kwargs={'max_rank': 20}, num_input_features = input_dim, 
                               num_classes = output_dim, hidden_size = hidden_size, LSTM_size=LSTM_size)
        swag_model.to(device)


        #loss and optimizer
        criterion = nn.CrossEntropyLoss() #crossentropy loss used for classification tasks
        #SGD with momentum and weight decay (weight_decay is 0)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr_init,weight_decay=weight_decay)
        Softmax = torch.nn.Softmax(dim=1)

        #learing rate scheduler, using custom swag lr scheduler, which decays the lr to the swag_lr
        learnrate_scheduler = swag_lr_scheduler(optimizer, lr_init, num_epochs, 
                                                swag = True, swag_start = swag_start, swag_lr = swag_lr, anneal_start = 0.1)
        #learnrate_scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.25)
        cyclic_learnrate_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=swag_lr, 
                                                                max_lr=lr_init, step_size_up = cyclic_multiplier*len(train_loader),
                                                                  cycle_momentum=False, mode = "triangular2")
        #cyclic = True
        swaganneal_learnrate_scheduler = torch.optim.swa_utils.SWALR(optimizer, 
                                                                     anneal_strategy="linear",anneal_epochs=2,swa_lr=swag_lr)


        #optional load model:
        """
        something like this:
        if args.resume is not None:
            print('Resume training from %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])"""
        #possible load swag to add here
        """
        something like this:
        if args.swag and args.swag_resume is not None:
            checkpoint = torch.load(args.swag_resume)
            swag_model.subspace.rank = torch.tensor(0)
            swag_model.load_state_dict(checkpoint['state_dict'])
        """

        print(f"Starting training on model {manyrunning}")
        # Train the model
        running_loss = 0.0
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            running_loss = 0.0
            #print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            for i, (traj, label) in enumerate(train_loader):
                traj = traj.to(device)
                label = label.to(device)

                # Forward pass
                output = model(traj)
                loss = criterion(output, label.view(-1))

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #calc accumulated loss
                running_loss += loss.item()

                if epoch+1 >= swag_start and (i+1)%int(n_total_steps/swag_update_freq) == 0:
                    #print("updating SWAG estimate")
                    swag_model.collect_model(model)

                if (i+1) % int(n_total_steps/5) == 0:
                    #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {running_loss/200:.4f}')

                    ############## TENSORBOARD ########################
                    #adding scalars for plots of loss function or similar
                    #add_scalar('name', value (y-axis), iteration number (x-axis))
                    #tensorboard smoothes the values, the opaque line is the unsmoothed one!
                    writer.add_scalar('training loss', running_loss / int(n_total_steps/5), epoch * n_total_steps + i)
                    running_loss = 0.0
                    ###################################################
                if cyclic == True and epoch+1 < swag_start:
                    cyclic_learnrate_scheduler.step()
            if epoch + 1 < swag_start:
                if cyclic == False:
                    learnrate_scheduler.step()
            else:
                swaganneal_learnrate_scheduler.step()

            #short test after each epoch
            with torch.no_grad():
                n_test_steps = len(test_loader)
                n_correct = 0
                n_samples = 0
                acc_loss = 0
                for traj, labels in test_loader:
                    traj = traj.to(device)
                    labels = labels.to(device)
                    if epoch+1 >= swag_start:
                        number_mc_samples = 20
                        output_samples = torch.ones(number_mc_samples, len(traj), output_dim, dtype=torch.float32).to(device)
                        output_samples_prob = torch.ones(number_mc_samples, len(traj), output_dim, dtype=torch.float32).to(device)
                        for j in range(number_mc_samples):
                            swag_model.sample()
                            output_samples[j] = swag_model(traj)
                            output_samples_prob[j] = Softmax(output_samples[j])

                        outputs = output_samples.mean(0)
                        outputs_prob = output_samples_prob.mean(0)
                        #swag_model.set_swa()
                        #outputs = swag_model(traj)
                    else:
                        outputs = model(traj)
                        outputs_prob = Softmax(outputs)

                    acc_loss += criterion(outputs, labels.view(-1)).item()

                    _, predicted = torch.max(outputs_prob.data, 1)
                    n_samples += labels.size(0)
                    n_correct += (predicted.view(-1) == labels.view(-1)).sum().item()

                accuracy = n_correct/n_samples
                #print(n_correct,n_samples)
                mean_loss = acc_loss/n_test_steps
                #print(f'Accuracy of the network on the 10000 test trajectories: {accuracy*100}%')

                ############## TENSORBOARD ########################
                #adding scalars for plots of loss function or similar
                #add_scalar('name', value (y-axis), iteration number (x-axis))
                #tensorboard smoothes the values, the opaque line is the unsmoothed one!
                writer.add_scalar('testing accuracy', accuracy, epoch * n_total_steps)
                writer.add_scalar('testing loss', mean_loss, epoch * n_total_steps)
                ###################################################


        # In[ ]:


        #test swag model
        number_mc_samples = 20
        classes = ['attm', 'ctrw', 'fbm', 'lw', 'sbm']

        with torch.no_grad():
            n_test_steps = len(test_loader)
            n_correct = 0
            n_samples = 0
            acc_loss = 0
            #acc_pred_var = 0
            n_class_truepositive = np.zeros(output_dim)
            n_class_falsepositive = np.zeros(output_dim)
            n_class_falsenegative = np.zeros(output_dim)
            conf_matrix = np.zeros((output_dim,output_dim))

            for traj, labels in test_loader:
                traj = traj.to(device)
                labels = labels.to(device)

                output_samples = torch.ones(number_mc_samples, len(traj), output_dim, dtype=torch.float32).to(device) 
                output_samples_prob = torch.ones(number_mc_samples, len(traj), output_dim, dtype=torch.float32).to(device)

                for i in range(number_mc_samples):
                    swag_model.sample()
                    output_samples[i] = swag_model(traj)
                    output_samples_prob[i] = Softmax(output_samples[i])

                outputs = output_samples.mean(0)
                outputs_prob = output_samples_prob.mean(0)
                #outputs_var = output_samples.var(0)

                #acc_pred_var += outputs_var.sum().item()


                acc_loss += criterion(outputs, labels.view(-1)).item()

                _, predicted = torch.max(outputs_prob.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted.view(-1) == labels.view(-1)).sum().item()


                for i in range(len(traj)): #determine number of true/false negative/positives
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_truepositive[label] += 1
                    else:
                        n_class_falsepositive[pred] += 1
                        n_class_falsenegative[label] += 1
                    conf_matrix[pred,label] += 1

            accuracy = n_correct/n_samples
            mean_loss = acc_loss/n_test_steps
            #mean_pred_var = acc_pred_var/n_samples
            print(f'Accuracy of the network on the 10000 test trajectories: {accuracy*100}%')
            print(f'Mean loss is: {mean_loss}')
            #print(f'Mean Variance predicted by SWAG is: {mean_pred_var}')

            class_precision = n_class_truepositive/(n_class_truepositive+n_class_falsepositive)
            class_recall = n_class_truepositive/(n_class_truepositive+n_class_falsenegative)
            class_f1_score = n_class_truepositive/(n_class_truepositive+0.5*(n_class_falsepositive+n_class_falsenegative))

            for i in range(output_dim):
                print(f"F1 score of class {classes[i]} is {class_f1_score[i]}")
            print(f"Mean F1 score is {class_f1_score.mean()}")

        """
        #plot confusion matrix
        import seaborn as sn
        import pandas as pd
        import matplotlib.pyplot as plt

        conf_matrix = conf_matrix/conf_matrix.sum(axis=0)


        df_cm = pd.DataFrame(conf_matrix, index = [i for i in classes],
                          columns = [i for i in classes])
        # plt.figure(figsize=(7,7))
        #sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True)

        plt.show()
        """

        # In[ ]:


        #saving

        #get time and date
        import time#, os, fnmatch, shutil
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)

        #save standardmodel
        name = "modelcheckpoint_" + timestamp
        directory = f"saves/classi/{dim}d/{T}_lenght/"
        path = directory + name

        import os
        try:
            os.mkdir(directory)
        except:
            print("directory exists")


        torch.save(model.state_dict(),path)

        #save swag model
        name = "swag_modelcheckpoint_" + timestamp
        directory = f"saves/classi/{dim}d/{T}_lenght/"
        path = directory + name

        torch.save(swag_model.state_dict(),path)

