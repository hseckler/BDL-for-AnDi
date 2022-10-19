# [Bayesian deep learning for error estimation in the analysis of anomalous diffusion](https://doi.org/10.21203/rs.3.rs-1902000/v1)



## Data
The data included in the */plotdata* directory features the results of applying trained *Multi SWAG* models on a randomly generated test data set. 
The file names correspond to different trajectory lengths, tasks or models. 
All data files include the true/predicted values for each trajectory in the test set in the following order:
- regression: [true exponent, predicted exponent, predicted standard deviation, true model, true noise]
- classification: [true model, confidence model 1, ..., confidence model 5, true exponent, true noise]
See the *nice_plotting.ipynb* and the *tkinter_evaluating.ipynb* files for use examples.

## Code
The code implements, trains and evaluates the *Multi SWAG* models. For the implementation of SWAG we use the code in the */swag* directory by [Pavel Izmailov](https://github.com/izmailovpavel/understandingbdl). Data sets are generated using the code in the */andi-code* directory by [Gorka Muñoz](https://github.com/AnDiChallenge/andi_datasets). See the *LICENSE* files in the corresponding directories.

In the main directory one may find the following files:
- ***LSTM_Neural_Network.py***: implementation of the neural network architectures used for the models
- ***swag_lr_scheduler.py***: custom learning rate scheduler used in training
- ***load_andi_dataset.py***: different classes for creating and loading datasets from saved files
- ***create_andi_datasets.ipynb***: used for creating dataset files loaded in *load_andi_dataset.py*, note that later some datasets mainly use the saved trajectories feature of the *andi datasets* package
- ***regression_run_aleatoric_uncertainty-superversion-manyrun.py***: training process for the regression of the anomalous exponent, running multiple times for different trajectory lengths
- ***regression_run_manyrun-singlemodel.py***: training process for the regression of the anomalous exponent with datasets containing only a single model, running multiple times for different trajectory lengths and all models
- ***classification_run-superversion-manyrun.py***: training process for the classification of the diffusion model, running multiple times for different trajectory lengths
- ***detailed_evaluate_regression.ipynb***: application of the trained models on a test data sets, obtained results are saved in *plotdata/*
- ***detailed_evaluate_regression_singlemodel.ipynb***: application of the trained models on a test data sets, obtained results are saved in *plotdata/*
- ***detailed_evaluate_classification.ipynb***: application of the trained models on a test data sets, obtained results are saved in *plotdata/*
- ***nice_plotting.ipynb***: uses the data in *plotdata* to plot the results
- ***tkinter_evaluating.ipynb***: uses the data in *plotdata* for interactive plotting using tkinter
