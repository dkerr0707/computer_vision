#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// --- Hyperparameters ---
const int BATCH_SIZE = 64;
const int EPOCHS = 10;
const float LEARNING_RATE = 1e-3;

// --- Model ---
struct MNISTNet : torch::nn::Module {
    torch::nn::Sequential features, classifier;

    MNISTNet()
        : features(
              torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)),
              torch::nn::ReLU(),
              torch::nn::MaxPool2d(2),
              torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
              torch::nn::ReLU(),
              torch::nn::MaxPool2d(2)),
          classifier(
              torch::nn::Flatten(),
              torch::nn::Linear(64 * 7 * 7, 128),
              torch::nn::ReLU(),
              torch::nn::Dropout(0.25),
              torch::nn::Linear(128, 10))
    {
        register_module("features", features);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        return classifier->forward(features->forward(x));
    }
};

// --- Train one epoch ---
template <typename Loader>
std::pair<double, double> train_epoch(
    MNISTNet& model,
    Loader& loader,
    torch::optim::Adam& optimizer,
    torch::Device device)
{
    model.train();
    double total_loss = 0.0;
    int correct = 0, total = 0;

    for (auto& batch : loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, targets);
        loss.backward();
        optimizer.step();

        total_loss += loss.item().toDouble() * data.size(0);
        correct += output.argmax(1).eq(targets).sum().item().toInt();
        total += data.size(0);
    }
    return {total_loss / total, static_cast<double>(correct) / total};
}

// --- Eval one epoch ---
template <typename Loader>
std::pair<double, double> eval_epoch(
    MNISTNet& model,
    Loader& loader,
    torch::Device device)
{
    model.eval();
    torch::NoGradGuard no_grad;
    double total_loss = 0.0;
    int correct = 0, total = 0;

    for (auto& batch : loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto output = model.forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, targets);

        total_loss += loss.item().toDouble() * data.size(0);
        correct += output.argmax(1).eq(targets).sum().item().toInt();
        total += data.size(0);
    }
    return {total_loss / total, static_cast<double>(correct) / total};
}

int main() {
    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA)
        : torch::Device(torch::kCPU);
    std::cout << "Using device: " << device << "\n";

    // --- Data ---
    auto train_dataset = torch::data::datasets::MNIST("data")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto val_dataset = torch::data::datasets::MNIST("data", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(BATCH_SIZE));

    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), torch::data::DataLoaderOptions().batch_size(BATCH_SIZE));

    // --- Model + optimizer ---
    MNISTNet model;
    model.to(device);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(LEARNING_RATE));

    // --- Training loop ---
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        auto [train_loss, train_acc] = train_epoch(model, *train_loader, optimizer, device);
        auto [val_loss, val_acc] = eval_epoch(model, *val_loader, device);

        std::cout << std::fixed << std::setprecision(4)
                  << "Epoch " << std::setw(2) << epoch << "/" << EPOCHS
                  << "  train_loss=" << train_loss
                  << "  train_acc=" << train_acc
                  << "  val_loss=" << val_loss
                  << "  val_acc=" << val_acc << "\n";
    }

    torch::save(std::make_shared<MNISTNet>(model), "mnist_model_cpp.pt");
    std::cout << "\nTraining complete. Model saved to mnist_model_cpp.pt\n";
    return 0;
}
