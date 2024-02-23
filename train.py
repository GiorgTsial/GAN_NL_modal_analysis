from NL_cycleGAN import CycleGAN


def main():
    cycleGAN = CycleGAN("data/3_DOF_nonlinear_disps.csv", checkpoints_folder="checkpoints/cycleGAN_checkpoints_3DOF_nonlinear")
    cycleGAN.fit_model()


if __name__ == "__main__":
    main()
