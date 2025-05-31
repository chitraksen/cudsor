#include "tensor.cpp"

int const EPOCHS = 10000;
float const LEARNING_RATE = 0.001f;

// train neural net on XOR data
int main() {
    using T = float;

    Linear<T> layer1(2, 64);
    Linear<T> layer2(64, 128);
    Linear<T> layer3(128, 64);
    Linear<T> layer4(64, 1);

    auto input_data =
        Tensor<T>({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0}, {4, 2}, false);
    auto target_data = Tensor<T>({0.0, 1.0, 1.0, 0.0}, {4, 1}, false);

    for (int epoch = 0; epoch <= EPOCHS; ++epoch) {
        layer1.zero_grad();
        layer2.zero_grad();
        layer3.zero_grad();
        layer4.zero_grad();

        // Forward pass
        Tensor linear1_out = layer1.forward(input_data);
        Tensor hidden1 = linear1_out.relu();

        Tensor linear2_out = layer2.forward(hidden1);
        Tensor hidden2 = linear2_out.relu();

        Tensor linear3_out = layer3.forward(hidden2);
        Tensor hidden3 = linear3_out.relu();

        Tensor output = layer4.forward(hidden3);

        auto loss = mse_loss(output, target_data);

        if (epoch % 1000 == 0) {
            std::printf("Epoch %5d, Loss: %.5f\n", epoch, loss.data()[0]);
        }

        loss.backward();

        layer1.update_weights(LEARNING_RATE);
        layer2.update_weights(LEARNING_RATE);
        layer3.update_weights(LEARNING_RATE);
        layer4.update_weights(LEARNING_RATE);
    }

    return EXIT_SUCCESS;
}
