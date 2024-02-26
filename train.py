from NL_cycleGAN import NL_CycleGAN
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training_data", type=str, help="The path to the training data", required=True)
    parser.add_argument("-c", "--checkpoint_folder", type=str, help="The path to the model checkpoints", required=True)
    parser.add_argument("-d", "--hidden_dim", type=int, help="The size of the hidden layer of the cycleGAN model")
    args = parser.parse_args()
    cycleGAN = NL_CycleGAN(args.training_data, hidden_dim=args.hidden_dim, checkpoints_folder=args.checkpoint_folder)
    cycleGAN.fit_model()


if __name__ == "__main__":
    main()
