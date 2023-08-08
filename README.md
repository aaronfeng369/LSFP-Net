# LSFP-Net

**Zhao, He, et al., *A deep unrolled neural network for real-time MRI-guided brain intervention.***

## Requirements

We provide an requirements file requirements.txt. You can create a new conda environment or virtual environment and execute install the requirements using the following command.

```
pip install -r requirements.txt
```
Training the LSFP-Net usually needs large GPU memory. If you come across the OOM problem, please try to reduce the `niter` and `n_f` of the network, or reduce the number of coils by coil compression.


## Data
We provided a sample dataset "./Datasets/IMRI_Simu6_test.mat" for testing the code. 


## Training
The network can be easily trained by run "TRAIN_LSFP_Net_C3_simu3_adj2.py".

Model files will be stored in `./model`.


## Testing
We provided a trained model file in "./model/LSFP_Net_C3_B3_SPF10_FPG5_adj2/net_params_100.pkl". 

The network can be easily tested by run "TEST_LSFP_Net_C3_simu3_adj2.py".
