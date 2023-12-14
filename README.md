# LSFP-Net

He, Z., Zhu, YN., Chen, Y. et al. A deep unrolled neural network for real-time MRI-guided brain intervention. Nat Commun 14, 8257 (2023). https://doi.org/10.1038/s41467-023-43966-w

https://doi.org/10.5281/zenodo.10054900

## Requirements

We provide a requirements file requirements.txt. You can create a new conda environment or virtual environment and execute install the requirements using the following command.

```
pip install -r requirements.txt
```

## Data
We provided a sample dataset "./Datasets/IMRI_Simu6_test.mat" for testing the code. 
You can also download complete datasets from https://doi.org/10.6084/m9.figshare.24473092.


## Training
The network can be easily trained by run "TRAIN_LSFP_Net_C3_simu3_adj2.py".

Model files will be stored in `./model`.


## Testing
We provided a trained model file in "./model/LSFP_Net_C3_B3_SPF10_FPG5_adj2/net_params_100.pkl". 

The network can be easily tested by run "TEST_LSFP_Net_C3_simu3_adj2.py".
