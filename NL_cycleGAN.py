import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, subtract, Dot, multiply, Lambda
from keras.models import Model, load_model
from keras import regularizers
from keras.optimizers import Adam
import os
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class CycleGAN:
    def __init__(self, disps_filepath, hidden_dim=100, checkpoints_folder="checkpoints"):
        self.disps_filepath = disps_filepath
        self.hidden_dim = hidden_dim
        self.checkpoints_folder = checkpoints_folder

    def _create_generator(self, input_dim, hidden_size=100, regular_param=0):
        input_ = Input(shape=(input_dim,))
        gen_1 = Dense(hidden_size, kernel_initializer='he_uniform', activation="tanh",
                      kernel_regularizer=regularizers.l2(regular_param))(input_)

        gen_2 = Dense(input_dim, kernel_initializer='he_uniform', activation="linear",
                      kernel_regularizer=regularizers.l2(regular_param))(gen_1)
        g_model = Model(input_, gen_2)
        return g_model

    def _create_discriminator(self, input_dim, hidden_size=200):
        input_ = Input(shape=(input_dim,))
        d_1 = Dense(hidden_size, kernel_initializer='he_uniform', activation="tanh")(input_)
        d_2 = Dense(1, kernel_initializer='he_uniform', activation="sigmoid")(d_1)
        d_model = Model(input_, d_2)
        d_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0005, beta_1=0.5))
        return d_model

    def _create_supervised_model(self, g_model, input_dim):
        input_ = Input(shape=(input_dim,))
        out = g_model(input_)
        supervised_model = Model(input_, out)
        # define optimization algorithm configuration
        opt = Adam(lr=0.001, beta_1=0.5)
        supervised_model.compile(loss="mse", optimizer=opt)
        return supervised_model

    def _define_composite_model(self, g_model_1, d_model, g_model_2, input_dim):
        # ensure the model we're updating is trainable
        g_model_1.trainable = True

        # mark discriminator as not trainable
        d_model.trainable = False

        # mark other generator model as not trainable
        g_model_2.trainable = False

        # discriminator element
        input_gen = Input(shape=(input_dim, ))
        gen1_out = g_model_1(input_gen)
        output_d = d_model(gen1_out)

        # identity element
        input_id = Input(shape=(input_dim, ))

        # forward cycle
        output_f = g_model_2(gen1_out)

        # backward cycle
        gen2_out = g_model_2(input_id)
        output_b = g_model_1(gen2_out)

        # define model graph
        model = Model([input_gen, input_id], [output_d, output_f, output_b])

        # define optimization algorithm configuration
        opt = Adam(lr=0.001, beta_1=0.5)

        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['binary_crossentropy', 'mse', 'mse'], loss_weights=[1, 5, 5], optimizer=opt)
        return model

    def _define_orthogonality_assembly(self, generator, sample_size):
        # Define the input layers
        input_1 = Input(shape=(sample_size,))
        input_2 = Input(shape=(sample_size,))
        input_3 = Input(shape=(sample_size,))
        input_4 = Input(shape=(sample_size,))

        # First vector calculation
        # Latent point 1

        # first point in real space
        p1 = generator(input_1)
        # second point in real space
        p2 = generator(input_2)

        # Vector in real space
        v1 = subtract([p1, p2])

        # second vector calculation
        # first point in real space
        p3 = generator(input_3)
        # second point in real space
        p4 = generator(input_4)

        # Vector in real space
        v2 = subtract([p3, p4])

        # Model output is the dot product of the two vectors
        out = Dot(axes=1)([v1, v2])

        tensor_1 = Dot(axes=1)([v1, v1])
        tensor_2 = Dot(axes=1)([v2, v2])

        mult = multiply([tensor_1, tensor_2])

        mult_1 = Lambda(lambda x: K.sqrt(x))(mult)

        out_1 = Lambda(lambda x: x[0] / x[1])([out, mult_1])

        product_model = Model([input_1, input_2, input_3, input_4], out_1)
        product_model.compile(loss="mse", optimizer=Adam(lr=0.0005, beta_1=0.5))
        return product_model

    def generate_real_samples(self, dataset, n_samples):
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], n_samples)

        # retrieve selected images
        X = dataset[ix]

        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        return X, y

    def generate_latent_dataset(self, n_dofs, n_fake_samples):
        means = np.zeros(n_dofs)
        cov = np.eye(n_dofs) * 0.25
        modal_coords = np.random.multivariate_normal(means, cov, n_fake_samples)

        modal_coords = 2 * (modal_coords - np.min(modal_coords, axis=0)) / (
                np.max(modal_coords, axis=0) - np.min(modal_coords, axis=0)) - 1.0
        return modal_coords

    def generate_fake_samples(self, g_model, dataset):
        # generate fake instance
        X = g_model.predict(dataset)
        y = np.zeros(len(X))
        return X, y

    def generate_product_latent_points(self, latent_dim, latent_dataset, n_samples, epsilon=1e-3):
        points = latent_dataset

        points_1 = np.copy(points)
        points_2 = np.copy(points)
        points_3 = np.copy(points)
        points_4 = np.copy(points)

        derivative_indices_1 = np.random.randint(0, latent_dim, n_samples)
        derivative_indices_2 = np.random.randint(0, latent_dim, n_samples)

        for i, index in enumerate(derivative_indices_1):
            points_1[i, index] += epsilon
            points_2[i, index] -= epsilon

            if derivative_indices_2[i] != index:
                points_3[i, derivative_indices_2[i]] += epsilon
                points_4[i, derivative_indices_2[i]] -= epsilon
            else:
                new_index = (derivative_indices_2[i] + 1) % latent_dim
                points_3[i, new_index] += epsilon
                points_4[i, new_index] -= epsilon

        z_input = np.hstack((points_1, points_2, points_3, points_4))
        return z_input

    def generate_product_latent_points_2(self, latent_dim, latent_dataset, n_samples, epsilon=1e-3):
        ix = np.random.randint(0, latent_dataset.shape[0], n_samples)
        points = latent_dataset[ix]

        points_1 = np.copy(points)
        points_2 = np.copy(points)
        points_3 = np.copy(points)
        points_4 = np.copy(points)

        derivative_indices_1 = np.random.randint(0, latent_dim, n_samples)
        derivative_indices_2 = np.random.randint(0, latent_dim, n_samples)

        for i, index in enumerate(derivative_indices_1):
            points_1[i, index] += epsilon
            points_2[i, index] -= epsilon

            points_1[i, :index] = 0
            points_1[i, index+1:] = 0

            points_2[i, :index] = 0
            points_2[i, index + 1:] = 0

            if derivative_indices_2[i] != index:
                points_3[i, derivative_indices_2[i]] += epsilon
                points_4[i, derivative_indices_2[i]] -= epsilon

                points_3[i, :derivative_indices_2[i]] = 0
                points_3[i, derivative_indices_2[i] + 1:] = 0

                points_4[i, :derivative_indices_2[i]] = 0
                points_4[i, derivative_indices_2[i] + 1:] = 0

            else:
                new_index = (derivative_indices_2[i] + 1) % latent_dim
                points_3[i, new_index] += epsilon
                points_4[i, new_index] -= epsilon

                points_3[i, :new_index] = 0
                points_3[i, new_index + 1:] = 0

                points_4[i, :new_index] = 0
                points_4[i, new_index + 1:] = 0

        z_input = np.hstack((points_1, points_2, points_3, points_4))
        return z_input

    def generate_product_dataset(self, latent_dim, latent_dataset, n_samples):
        z_input = self.generate_product_latent_points_2(latent_dim, latent_dataset, n_samples)
        y = np.zeros(n_samples)
        return z_input, y

    def generate_supervised_dataset(self, latent_dim, n_samples):
        z_input = np.zeros((n_samples, latent_dim))
        y = np.zeros((n_samples, latent_dim))
        return z_input, y

    def _save_models(self, g_model_AtoB, g_model_BtoA, epoch):
        cwd = os.getcwd()
        folder_name = self.checkpoints_folder
        folder_path = os.path.join(cwd, folder_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        filename_1 = os.path.join(os.getcwd(), folder_path, "g_model_AtoB_" + str(epoch) + ".h5")
        filename_2 = os.path.join(os.getcwd(), folder_path,  "g_model_BtoA_" + str(epoch) + ".h5")
        g_model_AtoB.save(filename_1)
        g_model_BtoA.save(filename_2)

    def train(self, d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, orthogonality_model,
              dataset, latent_dataset, n_epochs=10000, n_batch=2048, eval_every_epoch=100):
        # define properties of the training run
        n_epochs, n_batch, = n_epochs, n_batch

        # unpack dataset A = real data, B = modal space
        trainA = dataset
        trainB = latent_dataset

        # manually enumerate epochs
        for i in range(n_epochs):
            # select a batch of real samples
            X_realA, y_realA = self.generate_real_samples(trainA, n_batch)
            X_realB, y_realB = self.generate_real_samples(trainB, n_batch)

            # generate a batch of fake samples
            X_fakeA, y_fakeA = self.generate_fake_samples(g_model_BtoA, X_realB)
            X_fakeB, y_fakeB = self.generate_fake_samples(g_model_AtoB, X_realA)

            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realB, X_realA])

            # update discriminator for A -> [real/fake]
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realA, X_realB])

            # update discriminator for B -> [real/fake]
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

            # impose orthogonality
            x, y = self.generate_product_dataset(dataset.shape[1], latent_dataset, n_batch * 2 ** 4)
            x = [x[:, :dataset.shape[1]], x[:, dataset.shape[1]:2 * dataset.shape[1]], x[:, 2 * dataset.shape[1]:3 * dataset.shape[1]],
                 x[:, 3 * dataset.shape[1]:]]
            loss_product = orthogonality_model.train_on_batch(x, y)

            # # Force (0..., 0) to match (0..., 0)
            # x, y = self.generate_supervised_dataset(dataset.shape[1], 2)
            # supervised_loss = supervised_model.train_on_batch(x, y)

            # summarize performance
            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f] product_loss[%.3f, %f]' % (
                    i + 1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2, loss_product,
                    math.acos(loss_product) / np.pi * 180))

            if (i+1) % eval_every_epoch == 0:
                self._save_models(g_model_AtoB=g_model_AtoB, g_model_BtoA=g_model_BtoA, epoch=i+1)

    def fit_model(self):
        dataset = np.genfromtxt(self.disps_filepath, delimiter=',')

        transient_point = dataset.shape[0] // 10
        dataset = dataset[transient_point:, :]

        samples_dim = dataset.shape[1]

        dataset = 2 * (dataset - np.min(dataset, axis=0)) / (
                np.max(dataset, axis=0) - np.min(dataset, axis=0)) - 1.0

        latent_dataset = self.generate_latent_dataset(samples_dim, dataset.shape[0])

        # generator: A -> B
        g_model_AtoB = self._create_generator(samples_dim, hidden_size=self.hidden_dim)

        # generator: B -> A
        g_model_BtoA = self._create_generator(samples_dim, hidden_size=self.hidden_dim)

        # discriminator: A -> [real/fake]
        d_model_A = self._create_discriminator(samples_dim, hidden_size=self.hidden_dim)

        # discriminator: B -> [real/fake]
        d_model_B = self._create_discriminator(samples_dim, hidden_size=self.hidden_dim)

        # composite: A -> B -> [real/fake, A]
        c_model_AtoB = self._define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, samples_dim)

        # composite: B -> A -> [real/fake, B]
        c_model_BtoA = self._define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, samples_dim)

        # orthogonality inductive bias model
        ortho_model = self._define_orthogonality_assembly(g_model_BtoA, samples_dim)

        # train models
        self.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, ortho_model, dataset, latent_dataset)

