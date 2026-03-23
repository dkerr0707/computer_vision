#include <torch/torch.h>
#include <iostream>
#include <iomanip>

// Must match the architecture used in train_mnist.cpp
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

int main(int argc, char* argv[]) {
    const std::string model_path = (argc > 1) ? argv[1] : "mnist_model_cpp.pt";
    const std::string data_path  = (argc > 2) ? argv[2] : "data";
    const int batch_size = 64;

    torch::Device device = torch::cuda::is_available()
        ? torch::Device(torch::kCUDA)
        : torch::Device(torch::kCPU);
    std::cout << "Using device: " << device << "\n";

    // Load model
    auto model = std::make_shared<MNISTNet>();
    try {
        torch::load(model, model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model from '" << model_path << "': " << e.what() << "\n";
        return 1;
    }
    model->to(device);
    model->eval();
    std::cout << "Loaded model from " << model_path << "\n\n";

    // Load MNIST test set
    auto test_dataset = torch::data::datasets::MNIST(
            data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size));

    // Run inference
    torch::NoGradGuard no_grad;
    int correct = 0, total = 0;

    for (auto& batch : *test_loader) {
        auto data    = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto logits  = model->forward(data);
        auto preds   = logits.argmax(1);

        correct += preds.eq(targets).sum().item().toInt();

        for (int i = 0; i < data.size(0); ++i) {
            int pred   = preds[i].item().toInt();
            int target = targets[i].item().toInt();
            std::cout << "  sample " << std::setw(5) << total + i
                      << "  label=" << target
                      << "  pred=" << pred
                      << (pred == target ? "" : "  WRONG")
                      << "\n";
        }

        total += data.size(0);
    }

    double accuracy = static_cast<double>(correct) / total;
    std::cout << "\nTest accuracy: " << correct << "/" << total
              << " = " << std::fixed << std::setprecision(4) << accuracy << "\n";
    return 0;
}
