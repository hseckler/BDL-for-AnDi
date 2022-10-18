#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


class swag_lr_scheduler():
    def __init__(self, optimizer, lr_init, num_epochs, swag = False, swag_start = None, 
                 swag_lr = 0.02, anneal_start = 0.33, anneal_end = 0.8):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.num_epochs = num_epochs
        self.swag = swag
        self.swag_start = swag_start
        self.swag_lr = swag_lr
        self.epoch = 0 #start at epoch 0
        self.anneal_start = anneal_start
        self.anneal_end = anneal_end
        
    def schedule(self, epoch):
        t = (epoch) / (self.swag_start if self.swag else self.num_epochs)
        lr_ratio = self.swag_lr / self.lr_init if self.swag else 0.01
        if t <= self.anneal_start:
            factor = 1.0
        elif t <= self.anneal_end:
            factor = 1.0 - (1.0 - lr_ratio) * (t - self.anneal_start) / (self.anneal_end-self.anneal_start)
        else:
            factor = lr_ratio
        return self.lr_init * factor
    
    def change_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def step(self):
        self.epoch += 1
        new_lr = self.schedule(self.epoch)
        self.change_lr(new_lr)
        
        return new_lr
