# Nonlinear Modal Analysis using GANs

Code for [On the application of generative adversarial networks for nonlinear modal analysis](https://www.sciencedirect.com/science/article/pii/S0888327021008189) ([arxiv](https://arxiv.org/abs/2203.01229)).

To use the model, save the n-DOF displacements or accelerations as a csv file in the `data` directory.
The csv file should correspond to a numpy array with dimensions [n_timesteps, n-DOF], where n_timesteps is the number of available timestep samples and n-DOF is the number of degrees of freedom of the system.

To train the model:
```bash
python train.py --training_data [path to the csv of the training data] --checkpoint_folder [path to save the model checkpoints] --hidden_dim [number of hidden neurons of the cycleGAN model]
```
For example:
```bash
python train.py --training_data data/3_DOF_nonlinear_disps.csv --checkpoint_folder checkpoints/cycleGAN_checkpoints_3DOF_nonlinear
```
