# Nonlinear Modal Analysis using GANs

Code for [On the application of generative adversarial networks for nonlinear modal analysis](https://www.sciencedirect.com/science/article/pii/S0888327021008189) ([arxiv](https://arxiv.org/abs/2203.01229)).

To use the model, save the n-DOF displacements or accelerations as a csv file in the `data` directory.
The csv file should correspond to a numpy array with dimensions [n_timesteps, n--DOF], where n_timesteps is the number of available timestep samples and n-DOF is the number of degrees of freedom of the system.

To train the model:
```bash
python train.py --training_data [path to the csv of the training data] --checkpoint_folder [path to save the model checkpoints] --hidden_dim [number of hidden neurons of the cycleGAN model]
```
For example:
```bash
python train.py --training_data data/3_DOF_nonlinear_disps.csv --checkpoint_folder checkpoints/cycleGAN_checkpoints_3DOF_nonlinear
```

To run a linear modal analysis (or a PCA equivalent in the case of equal masses):
```bash
python PCA_modal_analysis.py --testing_data [path to the csv of the testing data] --linear_data [path to save the linearly--decomposed data] -f [the sampling frequency (optional, default=100)]
```

For example:
```bash
python PCA_modal_analysis.py --testing_data data/3_DOF_nonlinear_disps_test_dataset.csv --linear_data 3_DOF_nonlinear_disps_linear_modal.csv -f 100
```

To test the existing checkpoints for a 3DOF model using the inner-product criterion and extract the PSDs of the modal coordinates, as well as the comparisons with the linear decompositions, the correlations and the reconstruction error:
```bash
python.exe .\test_3DOF.py -tr [path to the csv of the training data] -t [path to the csv of the testing data] -l [path to the linearly--decomposed data] -c [path to the model checkpoints]
```

For example:
```bash
python.exe .\test_3DOF.py -tr .\data\3_DOF_nonlinear_disps.csv -t .\data\3_DOF_nonlinear_disps_test_dataset.csv -l .\data\3_DOF_nonlinear_disps_linear_modal.csv -c .\checkpoints\cycleGAN_checkpoints_3DOF_nonlinear\
```

And for a 4DOF model:
```bash
python.exe .\test_4DOF.py -tr [path to the csv of the training data] -t [path to the csv of the testing data] -l [path to the linearly--decomposed data] -c [path to the model checkpoints]
```

For example:
```bash
python.exe .\test_4DOF.py -tr .\data\4_DOF_nonlinear_disps.csv -t .\data\4_DOF_nonlinear_disps_test_dataset.csv -l .\data\4_DOF_nonlinear_disps_linear_modal.csv -c .\checkpoints\cycleGAN_checkpoints_4DOF_nonlinear\
```

The `checkpoints` directory includes the trained models presented in the [paper](https://arxiv.org/abs/2203.01229) for the simulated 3 and 4 DOF systems.
The data for the [LANL 3-storey building](https://www.osti.gov/biblio/961604) are property of the Los Alamos National Laboratory and cannot be released, however, we encourage you to test the algorithm on your data.
