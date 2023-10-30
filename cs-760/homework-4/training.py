from utils import training_scratch, training_pytorch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scratch_accuracies = []
    torch_accuracies = []

    scratch_losses = []
    torch_losses = []

    scratch_labels = []
    torch_labels = []

    # Scratch training
    d1_values = [200, 300]
    learning_rates = [0.01, 0.1]
    batch_sizes = [32, 64]

    for d1 in d1_values:
        for lr in learning_rates:
            for bs in batch_sizes:
                accuracy, loss_values = training_scratch(d1=d1, learning_rate=lr, num_epochs=10, batch_size=bs)
                scratch_accuracies.append(accuracy)
                scratch_losses.append(loss_values)
                scratch_labels.append(f"Scratch d1={d1}, lr={lr}, bs={bs}")

    # # PyTorch training
    for d1 in d1_values:
        for lr in learning_rates:
            for bs in batch_sizes:
                accuracy, loss_values  = training_pytorch(d1=d1, learning_rate=lr, num_epochs=10, batch_size=bs)
                torch_accuracies.append(accuracy)
                torch_losses.append(loss_values)
                torch_labels.append(f"Pytorch d1={d1}, lr={lr}, bs={bs}")


    # # PyTorch training with different weight initialization
    # accuracy_zeros, loss_values_zeros = training_pytorch(d1=200, learning_rate=0.01, num_epochs=50, batch_size=64, weight_init="zeros")
    # accuracy_random, loss_values_random = training_pytorch(d1=200, learning_rate=0.01, num_epochs=50, batch_size=64, weight_init="random")

    # # Build plot for all losses
    plt.figure()
    for i in range(len(scratch_losses)):
        plt.plot(scratch_losses[i], label=scratch_labels[i])
    plt.legend()
    plt.title("Losses for Scratch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figures/losses_scratch.pdf", bbox_inches="tight")
    plt.clf()

    plt.figure()
    for i in range(len(torch_losses)):
        plt.plot(torch_losses[i], label=torch_labels[i])
    plt.legend()
    plt.title("Losses for Pytorch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("figures/losses_pytorch.pdf", bbox_inches="tight")
    plt.clf()

    # # Build plot for accuracies
    plt.figure()
    plt.plot(scratch_accuracies, label="scratch")
    plt.plot(torch_accuracies, label="torch")
    plt.legend()
    plt.title("Accuracies for Scratch vs Torch")
    plt.xlabel("Setting")
    plt.ylabel("Accuracy")
    plt.savefig("figures/accuracies_scratch_vs_torch.pdf", bbox_inches="tight")