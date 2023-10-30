from utils import training_scratch, training_pytorch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scratch_accuracies = []
    torch_accuracies = []

    scratch_accuracies.append(training_scratch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=32))
    scratch_accuracies.append(training_scratch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=64))
    scratch_accuracies.append(training_scratch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=128))
    scratch_accuracies.append(training_scratch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=32))
    scratch_accuracies.append(training_scratch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=64))
    scratch_accuracies.append(training_scratch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=128))
    scratch_accuracies.append(training_scratch(d1=200, learning_rate=0.05, num_epochs=10, batch_size=64))
    scratch_accuracies.append(training_scratch(d1=200, learning_rate=0.1, num_epochs=10, batch_size=64))

    torch_accuracies.append(training_pytorch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=32))
    torch_accuracies.append(training_pytorch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=64))
    torch_accuracies.append(training_pytorch(d1=200, learning_rate=0.01, num_epochs=10, batch_size=128))
    torch_accuracies.append(training_pytorch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=32))
    torch_accuracies.append(training_pytorch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=64))
    torch_accuracies.append(training_pytorch(d1=300, learning_rate=0.01, num_epochs=10, batch_size=128))
    torch_accuracies.append(training_pytorch(d1=200, learning_rate=0.05, num_epochs=10, batch_size=64))
    torch_accuracies.append(training_pytorch(d1=200, learning_rate=0.1, num_epochs=10, batch_size=64))

    plt.figure()
    plt.plot(scratch_accuracies, label="scratch")
    plt.plot(torch_accuracies, label="torch")
    plt.legend()
    plt.title("Accuracies for Scratch vs Torch")
    plt.xlabel("Setting")
    plt.ylabel("Accuracy")
    plt.savefig("figures/accuracies_scratch_vs_torch.pdf", bbox_inches="tight")