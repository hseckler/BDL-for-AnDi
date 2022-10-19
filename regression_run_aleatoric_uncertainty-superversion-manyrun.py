#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.optim import lr_scheduler
from swag.posteriors import swag as swag
import time

from load_andi_dataset import *
from LSTM_Neural_Network import *
from swag_lr_scheduler import *




#Ts = [10,25,50,100,250,500,999]#[10,25,50,100,250,500,999]
#Ts = [10,100,500]
Ts = [500]
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
    if T > 500:
        N_train = int(5e5)
    N_test = 10000
    use_increments = True
    N_save = [16000,16000,10000,16000,10000]
    
    #loading from saved trajectories, allows for only one dataset of trajectories usable for all trajectory lenghts
    if dim == 1:
        train_path = "datasets/trajectories/"
    else:
        train_path = f"datasets/trajectories/{dim}d/"
    train_dataset = AnDi_dataset_from_saved_trajs(path = train_path, task = 1, dim = dim, N_total = N_train,
                                                  T = T, N_save = N_save, 
                                                  use_increments = use_increments)
    test_path = train_path + "testset/"
    test_dataset = AnDi_dataset_from_saved_trajs(path = test_path, task = 1, dim = dim, N_total = N_test, 
                                                 T = T, N_save = 500, use_increments = use_increments)
    



    print(len(train_dataset),len(test_dataset))
    print("Finished loading data")


    for manyrunning in range(25):

        #randomly choose hyperparameters i want to observe
        swag_lr = 1e-4#5e-4#np.random.choice([1e-3,1e-4,5e-4])#np.random.choice([1e-3,1e-4,5e-4])#4e-4#np.random.choice([0.03,0.01,0.005])
        #lr_init = np.random.choice([1e-3,5e-4])
        cyclic = True

        # Device configuration, run on gpu if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyper-parameters 
        input_dim = dim # 1D input sequence
        LSTM_size = [128,128,64]#64
        output_dim = 2 #output size is 2, one for anomalous exponent, 1 for its log variance
        num_epochs = 65
        batch_size = 512

        #optimizer hyper-paras
        lr_init = 2e-3#2e-3 #initial learning rate
        momentum = 0.9 #contribution of earlier gradient to next gradient
        weight_decay = 1e-4 #contribution L2 norm of weights to loss
        cyclic_multiplier = 4 #multiplier for the time it takes for one half of a cyclic period: e.g. 2 episodes

        #parameter choices for swag
        swag_start = 55 #when to start swag epoches (needs to be smaller than num epochs)
        swag_update_freq = 10 #number of times swag estimate is updated per epoch(x updates per epoch)
        if T == 250:
            swag_start = 75#65
            num_epochs = 85#75
            swag_update_freq = 10
            swag_lr = 5e-5
        if T >= 500:
            swag_start = 75#65
            num_epochs = 85#75
            swag_update_freq = 10
            swag_lr = 5e-5
        if T == 999:
            swag_start = 150
            num_epochs = 170
            swag_update_freq = 5
            cyclic_multiplier = 6
            swag_lr = 1e-4
        #swag_lr = 0.005 #swag learning rate

        

        # In[4]:

        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, pin_memory=False, num_workers = 3)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=False, pin_memory=False, num_workers = 3)


        # In[6]:


        ############## TENSORBOARD ########################
        #to view the tensorboard you need to run "tensorboard --logdir=runs" in console 
        #while in the directory of this file and open the link!
        import sys
        import torch.nn.functional as F
        from torch.utils.tensorboard import SummaryWriter
        import os

        #specify path for tensorboard
        i = 0 
        while os.path.exists(f"runs/aleatoric/{dim}d/{T}lenght/final_andiregress_cyclicdecay_{swag_start}-{num_epochs}_{lr_init:.4f}-{swag_lr:.5f}_mom{momentum:.2f}_batch{batch_size}swag%s" % i):
            i += 1
        tensorboardpath = f"runs/aleatoric/{dim}d/{T}lenght/final_andiregress_cyclicdecay_{swag_start}-{num_epochs}_{lr_init:.4f}-{swag_lr:.5f}_mom{momentum:.2f}_batch{batch_size}swag%s" % i
        #overwritten path (for custom name)
        #tensorboardpath = "runs/{N_test}lenght/aleatoric/andiregress_swagrun_25-35_lr0.15-0.02"
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(tensorboardpath)
        ###################################################


        # In[7]:


        #this doesn't seem to want to work properly
        #new idea: introduce activation function that fixes outputs to sensible values! (both to between 0 and 2?)


        # In[8]:


        #define model
        model = LSTM_Regression_aleatoric(input_dim, output_dim=output_dim, LSTM_size=LSTM_size).to(device)

        #define swag model
        swag_model = swag.SWAG(LSTM_Regression_aleatoric, subspace_type = 'covariance', 
                               subspace_kwargs={'max_rank': 20}, num_input_features = input_dim, 
                               output_dim = output_dim, LSTM_size=LSTM_size)
        swag_model.to(device)


        #loss and optimizer
        criterion = torch.nn.GaussianNLLLoss()#my_gnll_loss #gaussian negative log likelihood loss
        #SGD with momentum and weight decay; or adam
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr_init,weight_decay=weight_decay)

        #learing rate scheduler, using custom swag lr scheduler, which decays the lr to the swag_lr
        learnrate_scheduler = swag_lr_scheduler(optimizer, lr_init, num_epochs, 
                                                swag = True, swag_start = swag_start, swag_lr = swag_lr, anneal_start = 0.01)
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


        # In[ ]:

        print(f"starting training on model {manyrunning}")
        # Train the model
        running_loss = 0.0
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            starttime = time.process_time()

            running_loss = 0.0
            #print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            for i, (traj,target) in enumerate(train_loader):
                traj = traj.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Forward pass
                output = model(traj)
                loss = criterion(output[:,0].view(-1,1), target, output[:,1].view(-1,1))
                #print(output)
                # Backward and optimize
                optimizer.zero_grad()
                #print(loss)
                loss.backward()
                optimizer.step()
                #calc accumulated loss
                running_loss += loss.item()

                if epoch+1 >= swag_start and (i+1)%int(n_total_steps/swag_update_freq) == 0:
                    #print("updating SWAG estimate")
                    swag_model.collect_model(model)

                if (i+1) % int(n_total_steps/5) == 0:
                    #print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {running_loss/200:.4f}')
                    #print(output)
                    ############## TENSORBOARD ########################
                    #adding scalars for plots of loss function or similar
                    #add_scalar('name', value (y-axis), iteration number (x-axis))
                    #tensorboard smoothes the values, the opaque line is the unsmoothed one!
                    writer.add_scalar('training loss', running_loss / int(n_total_steps/5), epoch * n_total_steps + i)
                    running_loss = 0.0
                    ###################################################
                if epoch+1 < swag_start and cyclic == True:
                    cyclic_learnrate_scheduler.step()
            if epoch+1 < swag_start:
                if cyclic == False:
                    learnrate_scheduler.step()
            else:
                swaganneal_learnrate_scheduler.step()
            #print("Time needed for epoch "+str(time.process_time()-starttime))
            #short test after each epoch
            with torch.no_grad():
                n_test_steps = len(test_loader)
                n_abserr = 0
                n_samples = 0
                acc_loss = 0
                for traj, targets in test_loader:
                    traj = traj.to(device)
                    targets = targets.to(device)
                    if epoch+1 >= swag_start:

                        number_mc_samples = 20
                        output_samples = torch.ones(number_mc_samples, len(traj), 1, dtype=torch.float32).to(device)
                        variance_samples = torch.ones(number_mc_samples, len(traj), 1, dtype=torch.float32).to(device)
                        for i in range(number_mc_samples):
                            swag_model.sample()
                            model_output = swag_model(traj)
                            output_samples[i] = model_output[:,0].view(-1,1)
                            variance_samples[i] = model_output[:,1].view(-1,1)

                        output_exp = output_samples.mean(0)
                        outputted_var = variance_samples.mean(0)
                        combined_var = output_samples.var(0) + outputted_var

                        outputs = torch.cat((output_exp,outputted_var),1)


                        """number_mc_samples = 20
                        output_samples = torch.ones(number_mc_samples, len(traj), 2, dtype=torch.float32).to(device)
                        for j in range(number_mc_samples):
                            swag_model.sample()
                            output_samples[j] = swag_model(traj)

                        outputs = output_samples.mean(0)"""
                        #swag_model.set_swa()
                        #outputs = swag_model(traj)
                    else:
                        outputs = model(traj)

                    acc_loss += criterion(outputs[:,0].view(-1,1), targets, outputs[:,1].view(-1,1)).item()
                    n_samples += targets.size(0)
                    n_abserr += (outputs[:,0].view(-1,1)-targets).abs().sum().item()

                MAE = n_abserr / n_samples
                mean_loss = acc_loss/n_test_steps
                #print(f'MAE of the network on the 10000 test trajectories: {MAE}')

                ############## TENSORBOARD ########################
                #adding scalars for plots of loss function or similar
                #add_scalar('name', value (y-axis), iteration number (x-axis))
                #tensorboard smoothes the values, the opaque line is the unsmoothed one!
                writer.add_scalar('testing MAE', MAE, epoch * n_total_steps)
                writer.add_scalar('testing loss', mean_loss, epoch * n_total_steps)
                ###################################################

        #todo:
        #calc MAE - done looking good
        #why adam so much better? -higher LR helped!


        # In[ ]:





        # In[ ]:


        #test swag model
        number_mc_samples = 20

        MSELoss = torch.nn.MSELoss()

        with torch.no_grad():
            n_test_steps = len(test_loader)
            n_abserr = 0
            n_samples = 0
            acc_loss = 0
            acc_pred_var = 0
            acc_mse = 0
            for traj, targets in test_loader:
                traj = traj.to(device)
                targets = targets.to(device)

                output_samples = torch.ones(number_mc_samples, len(traj), 1, dtype=torch.float32).to(device)
                variance_samples = torch.ones(number_mc_samples, len(traj), 1, dtype=torch.float32).to(device)
                for i in range(number_mc_samples):
                    swag_model.sample()
                    model_output = swag_model(traj)
                    output_samples[i] = model_output[:,0].view(-1,1)
                    variance_samples[i] = model_output[:,1].view(-1,1)

                outputs = output_samples.mean(0)
                outputted_var = variance_samples.mean(0)
                combined_var = output_samples.var(0) + outputted_var

                acc_pred_var += combined_var.sum().item()
                """if epoch+1 >= swag_start:
                    swag_model.set_swa()
                    outputs = swag_model(traj)
                else:
                    outputs = model(traj)"""

                acc_loss += criterion(outputs, targets, outputted_var).item()
                acc_mse += MSELoss(outputs,targets)
                n_samples += targets.size(0)
                n_abserr += (outputs-targets).abs().sum().item()

            MAE = n_abserr / n_samples
            mean_loss = acc_loss/n_test_steps
            mean_pred_var = acc_pred_var/n_samples
            mean_mse = acc_mse/n_test_steps
            print(f'MAE of the network on the 10000 test trajectories: {MAE}')
            print(f'Mean loss is: {mean_loss}')
            print(f'Mean Variance predicted by SWAG is: {mean_pred_var}')
            print(f'Mean Squared Error is: {mean_mse}')


        # In[ ]:


        #saving

        #get time and date
        import time#, os, fnmatch, shutil
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)

        #save standardmodel
        name = "modelcheckpoint_" + timestamp
        directory = f"saves/aleatoric/{dim}d/{T}_lenght/"
        if dim == 1:
            directory = f"saves/aleatoric/{T}_lenght/"
            
        path = directory + name

        import os
        try:
            os.mkdir(directory)
        except:
            print("directory exists")


        torch.save(model.state_dict(),path)

        #save swag model
        name = "swag_modelcheckpoint_" + timestamp
        directory = f"saves/aleatoric/{dim}d/{T}_lenght/"
        if dim == 1:
            directory = f"saves/aleatoric/{T}_lenght/"
        path = directory + name

        torch.save(swag_model.state_dict(),path)





