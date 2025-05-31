#include "tensor.cpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

const size_t EPOCHS = 200;
const size_t BATCH_SIZE = 64;
const float LEARNING_RATE = 0.001f;

// function to reverse bytes for big-endian MNIST format
uint32_t reverse_bytes(uint32_t value) {
    return ((value & 0xFF000000) >> 24) | ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) | ((value & 0x000000FF) << 24);
}

// load MNIST images
std::vector<std::vector<float>> load_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magic = reverse_bytes(magic);
    num_images = reverse_bytes(num_images);
    rows = reverse_bytes(rows);
    cols = reverse_bytes(cols);

    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file format");
    }

    std::vector<std::vector<float>> images(num_images,
                                           std::vector<float>(rows * cols));

    for (uint32_t i = 0; i < num_images; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            uint8_t pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] =
                static_cast<float>(pixel) / 255.0f; // normalize to [0,1]
        }
    }

    std::cout << "Loaded " << num_images << " images of size " << rows << "x"
              << cols << std::endl;
    return images;
}

// load MNIST labels
std::vector<uint8_t> load_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic = reverse_bytes(magic);
    num_labels = reverse_bytes(num_labels);

    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file format");
    }

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    std::cout << "Loaded " << num_labels << " labels" << std::endl;
    return labels;
}

// convert labels to one-hot encoding
std::vector<std::vector<float>>
labels_to_one_hot(const std::vector<uint8_t>& labels) {
    std::vector<std::vector<float>> one_hot(labels.size(),
                                            std::vector<float>(10, 0.0f));
    for (size_t i = 0; i < labels.size(); ++i) {
        one_hot[i][labels[i]] = 1.0f;
    }
    return one_hot;
}

// create mini-batch from data
void create_batch(const std::vector<std::vector<float>>& images,
                  const std::vector<std::vector<float>>& labels,
                  const std::vector<size_t>& indices, size_t batch_start,
                  size_t batch_size, Tensor<float>& batch_images,
                  Tensor<float>& batch_labels) {
    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = indices[batch_start + i];

        // image data
        for (size_t j = 0; j < 784; ++j) {
            batch_images.data_[i * 784 + j] = images[idx][j];
        }

        // label data
        for (size_t j = 0; j < 10; ++j) {
            batch_labels.data_[i * 10 + j] = labels[idx][j];
        }
    }
}

float calculate_accuracy(const std::vector<std::vector<float>>& images,
                         const std::vector<uint8_t>& labels,
                         Linear<float>& layer1, Linear<float>& layer2,
                         Linear<float>& layer3, size_t num_samples) {

    size_t correct = 0;
    size_t test_batch_size = 100;

    for (size_t start = 0; start < num_samples; start += test_batch_size) {
        size_t current_batch_size =
            std::min(test_batch_size, num_samples - start);

        Tensor<float> test_batch({current_batch_size, 784}, false);

        // Fill test batch
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < 784; ++j) {
                test_batch.data_[i * 784 + j] = images[start + i][j];
            }
        }

        // Forward pass
        Tensor<float> h1 = layer1.forward(test_batch);
        Tensor<float> a1 = h1.relu();
        Tensor<float> h2 = layer2.forward(a1);
        Tensor<float> a2 = h2.relu();
        Tensor<float> output = layer3.forward(a2);

        // Find predictions
        for (size_t i = 0; i < current_batch_size; ++i) {
            float max_val = output.data_[i * 10];
            size_t pred = 0;
            for (size_t j = 1; j < 10; ++j) {
                if (output.data_[i * 10 + j] > max_val) {
                    max_val = output.data_[i * 10 + j];
                    pred = j;
                }
            }
            if (pred == labels[start + i]) {
                correct++;
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>(num_samples);
}

int main() {
    try {
        std::cout << "Loading MNIST dataset..." << std::endl;

        // load training data
        auto train_images = load_mnist_images("data/train-images-idx3-ubyte");
        auto train_labels_raw =
            load_mnist_labels("data/train-labels-idx1-ubyte");
        auto train_labels = labels_to_one_hot(train_labels_raw);

        // load test data
        auto test_images = load_mnist_images("data/t10k-images-idx3-ubyte");
        auto test_labels_raw = load_mnist_labels("data/t10k-labels-idx1-ubyte");

        std::cout << "Dataset loaded successfully!" << std::endl;

        // network: 784 -> 256 -> 128 -> 10
        Linear<float> layer1(784, 256, true);
        Linear<float> layer2(256, 128, true);
        Linear<float> layer3(128, 10, true);

        // training parameters
        const size_t num_train_samples = train_images.size();
        const size_t batches_per_epoch = num_train_samples / BATCH_SIZE;

        std::cout << "Starting training..." << std::endl;
        std::cout << "Network: 784 -> 256 -> 128 -> 10" << std::endl;
        std::cout << "Batch size: " << BATCH_SIZE << std::endl;
        std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
        std::cout << "Epochs: " << EPOCHS << std::endl;
        std::cout << "Batches per epoch: " << batches_per_epoch << std::endl;

        // create shuffled indices for training
        std::vector<size_t> train_indices(num_train_samples);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::random_device rd;
        std::mt19937 gen(rd());

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();
            // shuffle training data each epoch
            std::shuffle(train_indices.begin(), train_indices.end(), gen);

            float epoch_loss = 0.0f;
            size_t num_batches = 0;

            for (size_t batch_start = 0;
                 batch_start + BATCH_SIZE <= num_train_samples;
                 batch_start += BATCH_SIZE) {

                // zero grads
                layer1.zero_grad();
                layer2.zero_grad();
                layer3.zero_grad();

                // batch tensors
                Tensor<float> batch_images({BATCH_SIZE, 784}, false);
                Tensor<float> batch_labels({BATCH_SIZE, 10}, false);

                // fill batch
                create_batch(train_images, train_labels, train_indices,
                             batch_start, BATCH_SIZE, batch_images,
                             batch_labels);

                // forward pass
                Tensor<float> h1 = layer1.forward(batch_images);
                Tensor<float> a1 = h1.relu();
                Tensor<float> h2 = layer2.forward(a1);
                Tensor<float> a2 = h2.relu();
                Tensor<float> output = layer3.forward(a2);

                // loss
                Tensor<float> loss = mse_loss(output, batch_labels);
                epoch_loss += loss.data_[0];
                num_batches++;

                // backprop
                loss.backward();

                // update weights
                layer1.update_weights(LEARNING_RATE);
                layer2.update_weights(LEARNING_RATE);
                layer3.update_weights(LEARNING_RATE);
            }

            // print progress
            if (epoch % 1 == 0 || epoch == EPOCHS - 1) {
                auto epoch_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> epoch_duration =
                    epoch_end_time - epoch_start_time;

                float avg_loss = epoch_loss / num_batches;
                std::printf(
                    "Epoch %3lu/%lu, Average Loss: %.6f, Time Taken: %.3fs",
                    epoch, EPOCHS, avg_loss, epoch_duration.count());

                // training accuracy on subset
                if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
                    float train_acc = calculate_accuracy(
                        train_images, train_labels_raw, layer1, layer2, layer3,
                        std::min(size_t(1000), num_train_samples));
                    float test_acc =
                        calculate_accuracy(test_images, test_labels_raw, layer1,
                                           layer2, layer3, test_images.size());
                    std::cout << ", Train Acc: " << train_acc * 100 << "%"
                              << ", Test Acc: " << test_acc * 100 << "%";
                }
                std::cout << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time);

        std::cout << "\nTraining completed in " << duration.count()
                  << " seconds!" << std::endl;

        // final test/train acc
        std::cout << "\nFinal Evaluation:" << std::endl;
        float final_train_acc = calculate_accuracy(
            train_images, train_labels_raw, layer1, layer2, layer3,
            std::min(size_t(5000), num_train_samples));
        float final_test_acc =
            calculate_accuracy(test_images, test_labels_raw, layer1, layer2,
                               layer3, test_images.size());

        std::cout << "Final Training Accuracy: " << final_train_acc * 100 << "%"
                  << std::endl;
        std::cout << "Final Test Accuracy: " << final_test_acc * 100 << "%"
                  << std::endl;

        // example predictions
        std::cout << "\nSample Predictions:" << std::endl;
        for (int i = 0; i < 20; ++i) {
            Tensor<float> single_image({1, 784}, false);
            for (size_t j = 0; j < 784; ++j) {
                single_image.data_[j] = test_images[i][j];
            }

            Tensor<float> h1 = layer1.forward(single_image);
            Tensor<float> a1 = h1.relu();
            Tensor<float> h2 = layer2.forward(a1);
            Tensor<float> a2 = h2.relu();
            Tensor<float> output = layer3.forward(a2);

            float max_val = output.data_[0];
            size_t pred = 0;
            for (size_t j = 1; j < 10; ++j) {
                if (output.data_[j] > max_val) {
                    max_val = output.data_[j];
                    pred = j;
                }
            }

            std::cout << "Sample " << i
                      << ": True=" << static_cast<int>(test_labels_raw[i])
                      << ", Predicted=" << pred << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
