# BDL-for-AnDi
[Bayesian deep learning for error estimation in the analysis of anomalous diffusion](https://doi.org/10.21203/rs.3.rs-1902000/v1)


## Data
The data included in the *plotdata/* directory features the results of applying trained *Multi SWAG* models on a randomly generated test data set. 
The file names correspond to different trajectory lengths, tasks or models. 
All data files include the true/predicted values for each trajectory in the test set in the following order:
- regression: [true exponent, predicted exponent, predicted standard deviation, true model, true noise]
- classification: [true model, confidence model 1, ..., confidence model 5, true exponent, true noise]
