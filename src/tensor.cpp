#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

// forward declarations
template <typename T> class Tensor;
template <typename T> class AutogradNode;
template <typename T> class MSENode;
template <typename T> class LinearNode;

// base class for autograd nodes
template <typename T> class AutogradNode {
  public:
    virtual ~AutogradNode() = default;
    // grad_output is the gradient of the loss with respect to the output of
    // this node's operation
    virtual void backward(const std::vector<T>& grad_output) = 0;
    std::vector<Tensor<T>*> inputs;
};

// specific autograd nodes
template <typename T> class AddNode : public AutogradNode<T> {
  public:
    void backward(const std::vector<T>& grad_from_downstream) override {
        // Y = A + B. dL/dA = dL/dY * 1; dL/dB = dL/dY * 1
        // grad_from_downstream is dL/dY and has the same shape as A and B.

        if (this->inputs.size() < 2)
            return;

        Tensor<T>* input0 = this->inputs[0];
        Tensor<T>* input1 = this->inputs[1];

        if (input0 != nullptr && input0->requires_grad_) {
            // dL/d(input0) same as grad_from downstream
            for (size_t i = 0; i < input0->grad_.size(); ++i) {
                if (i < grad_from_downstream.size()) {
                    input0->grad_[i] += grad_from_downstream[i];
                }
            }
            if (input0->grad_fn_) {
                input0->grad_fn_->backward(grad_from_downstream);
            }
        }

        if (input1 != nullptr && input1->requires_grad_) {
            // dL/d(input1) same as grad_from downstream
            for (size_t i = 0; i < input1->grad_.size(); ++i) {
                if (i < grad_from_downstream.size()) {
                    input1->grad_[i] += grad_from_downstream[i];
                }
            }
            if (input1->grad_fn_) {
                input1->grad_fn_->backward(grad_from_downstream);
            }
        }
    }
};

template <typename T> class MulNode : public AutogradNode<T> {
  public:
    void backward(const std::vector<T>& grad_from_downstream) override {
        // Y = A * B. dL/dA = dL/dY * B; dL/dB = dL/dY * A
        if (this->inputs.size() < 2)
            return;

        Tensor<T>* input0 = this->inputs[0];
        Tensor<T>* input1 = this->inputs[1];

        if (input0 != nullptr && input0->requires_grad_) {
            std::vector<T> grad_for_input0(input0->numel());
            // dL/d(input0)
            for (size_t i = 0; i < grad_for_input0.size(); ++i) {
                if (i < grad_from_downstream.size() &&
                    i < input1->data_.size()) {
                    grad_for_input0[i] =
                        grad_from_downstream[i] * input1->data_[i];
                    input0->grad_[i] += grad_for_input0[i];
                }
            }
            if (input0->grad_fn_) {
                input0->grad_fn_->backward(grad_for_input0);
            }
        }

        if (input1 != nullptr && input1->requires_grad_) {
            std::vector<T> grad_for_input1(input1->numel());
            // dL/d(input1)
            for (size_t i = 0; i < grad_for_input1.size(); ++i) {
                if (i < grad_from_downstream.size() &&
                    i < input0->data_.size()) {
                    grad_for_input1[i] =
                        grad_from_downstream[i] * input0->data_[i];
                    input1->grad_[i] += grad_for_input1[i];
                }
            }
            if (input1->grad_fn_) {
                input1->grad_fn_->backward(grad_for_input1);
            }
        }
    }
};

template <typename T> class MatMulNode : public AutogradNode<T> {
  public:
    // store shapes for backward pass if needed, or retrieve from input tensors
    // Y = A @ B. A is (m,k), B is (k,n), Y is (m,n)
    // dL/dA = dL/dY @ B^T
    // dL/dB = A^T @ dL/dY
    void backward(const std::vector<T>& grad_from_downstream_flat) override {
        if (this->inputs.size() < 2)
            return;

        Tensor<T>* A_tensor = this->inputs[0]; // (m, k)
        Tensor<T>* B_tensor = this->inputs[1]; // (k, n)

        size_t m = A_tensor->shape_[0];
        size_t n = B_tensor->shape_[1];

        // reshape grad_from_downstream_flat (dL/dY) to (m, n)
        Tensor<T> downstream_grad_mat(grad_from_downstream_flat, {m, n}, false);

        if (A_tensor != nullptr && A_tensor->requires_grad_) {
            // dL/dA = dL/dY @ B^T
            // B_tensor shape: (k, n), B_tensor->transpose() shape: (n, k)
            // downstream_grad_mat shape: (m, n)
            // grad_A_tensor shape: (m, n) @ (n, k) -> (m, k)
            Tensor<T> B_transposed = B_tensor->transpose();
            Tensor<T> grad_A_tensor = downstream_grad_mat.matmul(B_transposed);
            A_tensor->accumulate_grad(grad_A_tensor);

            if (A_tensor->grad_fn_) {
                A_tensor->grad_fn_->backward(grad_A_tensor.data_);
            }
        }

        if (B_tensor != nullptr && B_tensor->requires_grad_) {
            // dL/dB = A^T @ dL/dY
            // A_tensor shape: (m, k), A_tensor->transpose() shape: (k, m)
            // downstream_grad_mat shape: (m, n)
            // grad_B_tensor shape: (k, m) @ (m, n) -> (k, n)
            Tensor<T> A_transposed = A_tensor->transpose();
            Tensor<T> grad_B_tensor = A_transposed.matmul(downstream_grad_mat);
            B_tensor->accumulate_grad(grad_B_tensor);

            if (B_tensor->grad_fn_) {
                B_tensor->grad_fn_->backward(grad_B_tensor.data_);
            }
        }
    }
};

template <typename T> class SumNode : public AutogradNode<T> {
  public:
    void backward(const std::vector<T>& grad_from_downstream) override {
        // dL/d(ScalarOutput of Sum)
        if (this->inputs.empty())
            return;

        Tensor<T>* input_tensor = this->inputs[0];

        if (input_tensor != nullptr && input_tensor->requires_grad_) {
            T dL_dS = grad_from_downstream[0];

            // dL/d(I_j) = (dL/dS) * (dS/d(I_j)) = dL_dS * 1
            std::vector<T> input_grad(input_tensor->numel());

            for (size_t i = 0; i < input_tensor->grad_.size(); ++i) {
                input_grad[i] = dL_dS;
                input_tensor->grad_[i] += dL_dS;
            }

            if (input_tensor->grad_fn_) {
                input_tensor->grad_fn_->backward(input_grad);
            }
        }
    }
};

template <typename T> class ReLUNode : public AutogradNode<T> {
  public:
    void backward(const std::vector<T>& grad_from_downstream) override {
        // Y = max(0, X). dY/dX = 1 if X > 0, else 0.
        // dL/dX = dL/dY * (dY/dX)
        if (this->inputs.empty())
            return;

        Tensor<T>* input_tensor = this->inputs[0]; // X

        if (input_tensor != nullptr && input_tensor->requires_grad_) {
            std::vector<T> input_grad(input_tensor->numel());

            for (size_t i = 0; i < input_tensor->grad_.size(); ++i) {
                if (i < grad_from_downstream.size()) {
                    T derivative = (input_tensor->data_[i] > 0) ? T(1) : T(0);
                    input_grad[i] = grad_from_downstream[i] * derivative;
                    input_tensor->grad_[i] += input_grad[i];
                }
            }
            if (input_tensor->grad_fn_) {
                input_tensor->grad_fn_->backward(input_grad);
            }
        }
    }
};

template <typename T> class BiasAddNode : public AutogradNode<T> {
  public:
    // Y = X + b (b is broadcast over X)
    // X is (batch_size, features), b is (features,)
    // dL/dX = dL/dY * 1 (element-wise)
    // dL/db_j = sum over batch_dim (dL/dY_ij)

    size_t batch_size;
    size_t features;

    void backward(const std::vector<T>& grad_from_downstream_flat) override {
        if (this->inputs.size() < 2)
            return;

        Tensor<T>* input_X = this->inputs[0]; // X (batch_size, features)
        Tensor<T>* bias_b = this->inputs[1];  // b (features,)

        // grad_from_downstream_flat has shape (batch_size * features)
        Tensor<T> grad_output_mat(grad_from_downstream_flat,
                                  {this->batch_size, this->features}, false);

        if (input_X != nullptr && input_X->requires_grad_) {
            // dL/dX is element-wise dL/dY
            // grad_from_downstream_flat is already dL/dX in flat form
            input_X->accumulate_grad(grad_output_mat);

            if (input_X->grad_fn_) {
                // pass the flat gradient dL/dX
                input_X->grad_fn_->backward(grad_from_downstream_flat);
            }
        }

        if (bias_b != nullptr && bias_b->requires_grad_) {
            // dL/db_j = sum over batch_dim (dL/dY_ij)
            // can be computed as ones_row_vec @ grad_output_mat
            // ones_row_vec (1, batch_size), grad_output_mat (batch_size,
            // features) result (1, features)
            Tensor<T> ones_row_vec({1, this->batch_size});
            ones_row_vec.fill(T(1));

            Tensor<T> grad_b_tensor_row =
                ones_row_vec.matmul(grad_output_mat); // (1, features)

            std::vector<T> grad_for_b(this->features);

            for (size_t i = 0; i < this->features; ++i) {
                grad_for_b[i] = grad_b_tensor_row.data_[i];
                bias_b->grad_[i] += grad_b_tensor_row.data_[i];
            }

            if (bias_b->grad_fn_) {
                bias_b->grad_fn_->backward(grad_for_b);
            }
        }
    }
};

// Main Tensor class
template <typename T> class Tensor {
  public:
    std::vector<T> data_;
    std::vector<T> grad_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    bool requires_grad_;
    std::shared_ptr<AutogradNode<T>> grad_fn_;

    void calculate_strides() {
        strides_.resize(shape_.size());
        if (!shape_.empty()) {
            strides_.back() = 1;
            for (int i = shape_.size() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }
        }
    }

    size_t calculate_total_size() const {
        if (shape_.empty())
            return 0;
        return std::accumulate(shape_.begin(), shape_.end(), size_t(1),
                               std::multiplies<size_t>());
    }

  public:
    Tensor() : requires_grad_(false) {}

    // init with data and shape
    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape,
           bool requires_grad = false)
        : data_(data), shape_(shape), requires_grad_(requires_grad) {
        calculate_strides();
        size_t total_size = calculate_total_size();
        assert(data_.size() == total_size &&
               "Data size must match shape product.");
        if (requires_grad_) {
            grad_.resize(total_size, T(0));
        }
    }

    // init with only shape - fills data 0
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false)
        : shape_(shape), requires_grad_(requires_grad) {
        calculate_strides();
        size_t total_size = calculate_total_size();
        data_.resize(total_size, T(0));
        if (requires_grad_) {
            grad_.resize(total_size, T(0));
        }
    }

    // avoids unnecessary copies
    Tensor(std::vector<T>&& data, const std::vector<size_t>& shape,
           bool requires_grad = false)
        : data_(std::move(data)), shape_(shape), requires_grad_(requires_grad) {
        calculate_strides();
        assert(data_.size() == calculate_total_size());
        if (requires_grad_) {
            grad_.resize(data_.size(), T(0));
        }
    }

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    bool empty() const { return data_.empty(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t numel() const { return data_.size(); }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    T* grad_data() { return grad_.data(); }
    const T* grad_data() const { return grad_.data(); }
    bool requires_grad() const { return requires_grad_; }

    size_t linear_index(const std::vector<size_t>& indices) const {
        assert(indices.size() == shape_.size());
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape_[i]);
            idx += indices[i] * strides_[i];
        }
        return idx;
    }

    // nD indexing
    T& operator()(const std::vector<size_t>& indices) {
        return data_[linear_index(indices)];
    }
    T operator()(const std::vector<size_t>& indices) const {
        return data_[linear_index(indices)];
    }

    // 2D indexing
    T& operator()(size_t i, size_t j) {
        assert(shape_.size() == 2);
        return data_[i * shape_[1] + j];
    }
    T operator()(size_t i, size_t j) const {
        assert(shape_.size() == 2);
        return data_[i * shape_[1] + j];
    }

    // 1D indexing
    T& operator()(size_t i) {
        assert(shape_.size() == 1);
        return data_[i];
    }
    T operator()(size_t i) const {
        assert(shape_.size() == 1);
        return data_[i];
    }

    // file tensor data with constant value
    Tensor& fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
        return *this;
    }

    // fill tensor data with normal values
    Tensor& random_normal(T mean = 0, T std = 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, std);
        for (T& val : data_)
            val = dist(gen);
        return *this;
    }

    // fill tensor with xavier uniform values - only works for 2D tensors atm
    Tensor& xavier_uniform() {
        if (shape_.size() != 2)
            return *this; // else throw error?

        T fan_in = shape_[1];
        T fan_out = shape_[0];
        T bound = std::sqrt(6.0 / (fan_in + fan_out));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-bound, bound);
        for (T& val : data_)
            val = dist(gen);
        return *this;
    }

    // operations
    Tensor operator+(Tensor& other) {
        assert(shape_ == other.shape_ &&
               "Tensor shapes must match for element-wise addition.");
        bool result_requires_grad = requires_grad_ || other.requires_grad_;

        Tensor result(shape_, result_requires_grad);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + other.data_[i];

        if (result_requires_grad) {
            auto node = std::make_shared<AddNode<T>>();
            node->inputs.push_back(this);
            node->inputs.push_back(&other);
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor operator*(Tensor& other) {
        assert(shape_ == other.shape_ &&
               "Tensor shapes must match for element-wise multiplication.");

        bool result_requires_grad = requires_grad_ || other.requires_grad_;
        Tensor result(shape_, result_requires_grad);

        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * other.data_[i];

        if (result_requires_grad) {
            auto node = std::make_shared<MulNode<T>>();
            node->inputs.push_back(this);
            node->inputs.push_back(&other);
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor operator+(T scalar) const {
        Tensor result = Tensor(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + scalar;

        // NOTE: autograd not implemented here. this will break chain if used
        // mid-graph with requires_grad.
        return result;
    }

    Tensor operator*(T scalar) const {
        Tensor result = Tensor(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * scalar;

        // NOTE: autograd not implemented here. this will break chain if used
        // mid-graph with requires_grad.
        return result;
    }

    Tensor broadcast_add(Tensor& other_bias) {
        // assuming this is (batch_size, features) + (features,)
        assert(shape_.size() == 2 && other_bias.shape_.size() == 1 &&
               "broadcast_add expects (batch,feat) + (feat)");
        assert(shape_[1] == other_bias.shape_[0] &&
               "Feature dimension mismatch in broadcast_add");

        size_t batch_size = shape_[0];
        size_t features = shape_[1];

        bool result_requires_grad = requires_grad_ || other_bias.requires_grad_;
        Tensor result(shape_, result_requires_grad);

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                result.data_[b * features + f] =
                    data_[b * features + f] + other_bias.data_[f];
            }
        }

        if (result_requires_grad) {
            auto node = std::make_shared<BiasAddNode<T>>();
            node->batch_size = batch_size;       // store for backward pass
            node->features = features;           // store for backward pass
            node->inputs.push_back(this);        // X
            node->inputs.push_back(&other_bias); // b
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor matmul(Tensor& other) {
        assert(shape_.size() == 2 && other.shape_.size() == 2 &&
               "Matmul inputs must be 2D tensors.");
        assert(shape_[1] == other.shape_[0] &&
               "Matrix dimensions incompatible for matmul.");

        size_t m = shape_[0];
        size_t k = shape_[1];
        size_t n = other.shape_[1];
        bool result_requires_grad = requires_grad_ || other.requires_grad_;
        Tensor result({m, n}, result_requires_grad);

        const T* a_data = data_.data();
        const T* b_data = other.data_.data();
        T* result_data = result.data_.data();

        // i-k-j loop order for better cache locality
        for (size_t i = 0; i < m; ++i) {
            for (size_t l = 0; l < k; ++l) {
                T a_val = a_data[i * k + l];         // direct access
                T* result_row = &result_data[i * n]; // cache row pointer
                const T* b_row = &b_data[l * n];
                for (size_t j = 0; j < n; ++j) {
                    result_row[j] += a_val * b_row[j];
                }
            }
        }

        if (result_requires_grad) {
            auto node = std::make_shared<MatMulNode<T>>();
            node->inputs.push_back(this);
            node->inputs.push_back(&other);
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor transpose() const {
        assert(shape_.size() == 2 &&
               "Transpose currently only supports 2D tensors.");
        Tensor result({shape_[1], shape_[0]}, requires_grad_);
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }

        // NOTE: autograd not implemented here. this will break chain if used
        // mid-graph with requires_grad.
        return result;
    }

    Tensor sum() {
        Tensor result({1}, requires_grad_); // scalar output
        T total = 0;
        for (const T& val : data_)
            total += val;
        result.data_[0] = total;

        // sum node created only if input requires grad
        if (requires_grad_) {
            auto node = std::make_shared<SumNode<T>>();
            node->inputs.push_back(this);
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor relu() {
        Tensor result(shape_, requires_grad_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = std::max(T(0), data_[i]);

        // relu node created only if input requires grad
        if (requires_grad_) {
            auto node = std::make_shared<ReLUNode<T>>();
            node->inputs.push_back(this);
            result.grad_fn_ = node;
        }
        return result;
    }

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                          size_t(1), std::multiplies<size_t>());
        assert(new_size == data_.size() &&
               "Total number of elements must remain the same for reshape.");

        // copy data, grad, requires_grad, grad_fn, and then change shape field,
        // stride info
        Tensor result = *this;
        result.shape_ = new_shape;
        result.calculate_strides();

        // NOTE: autograd not implemented here. this will break chain if used
        // mid-graph with requires_grad.
        return result;
    }

    void zero_grad() {
        if (requires_grad_) {
            std::fill(grad_.begin(), grad_.end(), T(0));
        }
    }

    void backward() {
        assert(data_.size() == 1 &&
               "Backward can only be called on a scalar tensor (loss).");
        if (!requires_grad_)
            return;

        // for the final loss tensor, dL/dL = 1.
        std::vector<T> initial_gradient = {T(1)};

        if (grad_fn_) {
            grad_fn_->backward(initial_gradient);
        }
    }

    // accumulates gradient into [grad_] if passed the grad update in same shape
    void accumulate_grad(const Tensor<T> contribution) {
        if (requires_grad_) {
            assert(contribution.shape_ == shape_);

            const T* contrib_data = contribution.data_.data();
            T* grad_data = grad_.data();

            for (size_t i = 0; i < grad_.size(); ++i) {
                grad_data[i] += contrib_data[i];
            }
        }
    }

    // print tensor for debugging
    void print() const {
        if (shape_.size() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << data_[i];
                if (i < shape_[0] - 1)
                    std::cout << ", ";
            }
            std::cout << "]\n";
        } else if (shape_.size() == 2) {
            std::cout << "[\n";
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << "  [";
                for (size_t j = 0; j < shape_[1]; ++j) {
                    std::cout << (*this)(i, j);
                    if (j < shape_[1] - 1)
                        std::cout << ", ";
                }
                std::cout << "]";
                if (i < shape_[0] - 1)
                    std::cout << ",";
                std::cout << "\n";
            }
            std::cout << "]\n";
        }
    }
};

// Neural Network Components
template <typename T> class LinearNode : public AutogradNode<T> {
  public:
    void backward(const std::vector<T>& grad_from_downstream_flat) override {
        if (this->inputs.size() < 3)
            return;

        Tensor<T>* layer_input =
            this->inputs[0];                 // X: (batch_size, in_features)
        Tensor<T>* weight = this->inputs[1]; // W: (out_features, in_features)
        Tensor<T>* bias = this->inputs[2];   // b: (out_features,)

        assert(layer_input->requires_grad_ || weight->requires_grad_ ||
               bias->requires_grad_ &&
                   "LinearNode backward called but no inputs require grad.");

        // get shapes from inputs
        size_t batch_size = layer_input->shape_[0];
        size_t out_features = weight->shape_[0];

        // reshape grad_from_downstream (dL/dY) to matrix
        Tensor<T> grad_output_mat(grad_from_downstream_flat,
                                  {batch_size, out_features}, false);

        // gradient w.r.t. weights: dL/dW = (dL/dY)^T @ X
        // grad_output_mat^T: (out_features, batch_size)
        // X: (batch_size, in_features)
        // dL/dW: (out_features, batch_size) @ (batch_size, in_features) ->
        // (out_features, in_features)
        if (weight->requires_grad_) {
            Tensor<T> grad_output_transposed = grad_output_mat.transpose();
            Tensor<T> grad_W_tensor =
                grad_output_transposed.matmul(*layer_input);
            weight->accumulate_grad(grad_W_tensor);

            if (weight->grad_fn_) {
                weight->grad_fn_->backward(grad_W_tensor.data_);
            }
        }

        // gradient w.r.t. bias: dL/db = sum over batch_dim (dL/dY)
        // ones_row_vec (1, batch_size) @ grad_output_mat (batch_size,
        // out_features) -> (1, out_features)
        if (bias->requires_grad_) {
            Tensor<T> ones_row_vec({1, batch_size});
            ones_row_vec.fill(T(1));
            Tensor<T> grad_b_tensor_row =
                ones_row_vec.matmul(grad_output_mat); // (1, out_features)

            // hold calculated gradient for propagation
            std::vector<T> grad_for_bias(out_features);

            for (size_t i = 0; i < out_features; ++i) {
                bias->grad_[i] += grad_b_tensor_row.data_[i];
                grad_for_bias[i] = grad_b_tensor_row.data_[i];
            }
            if (bias->grad_fn_) {
                bias->grad_fn_->backward(grad_for_bias);
            }
        }

        // gradient w.r.t. layer_input: dL/dX = dL/dY @ W
        // grad_output_mat: (batch_size, out_features)
        // W: (out_features, in_features)
        // dL/dX: (batch_size, out_features) @ (out_features, in_features) ->
        // (batch_size, in_features)
        if (layer_input->requires_grad_) {
            Tensor<T> grad_X_tensor = grad_output_mat.matmul(*weight);
            layer_input->accumulate_grad(grad_X_tensor);

            if (layer_input->grad_fn_) {
                layer_input->grad_fn_->backward(grad_X_tensor.data_);
            }
        }
    }
};

template <typename T> class Linear {
  public:
    Tensor<T> weight_;          // (out_features, in_features)
    Tensor<T> bias_;            // (out_features,)
    bool requires_grad_params_; // if weights/bias require grad

    Linear(size_t in_features, size_t out_features,
           bool requires_grad_params = true)
        : weight_({out_features, in_features}, requires_grad_params),
          bias_({out_features}, requires_grad_params),
          requires_grad_params_(requires_grad_params) {
        weight_.xavier_uniform();
        bias_.fill(0);
    }

    // forward pass
    Tensor<T> forward(Tensor<T>& input) {
        // input: (batch_size, in_features)
        // Y = input @ W^T + b
        Tensor<T> weight_transposed =
            weight_.transpose(); // (in_features, out_features)
        Tensor<T> matmul_result =
            input.matmul(weight_transposed); // (batch_size, out_features)
        Tensor<T> output =
            matmul_result.broadcast_add(bias_); // (batch_size, out_features)

        bool result_requires_grad =
            (input.requires_grad() || weight_.requires_grad() ||
             bias_.requires_grad());
        output.requires_grad_ = result_requires_grad;

        if (result_requires_grad) {
            auto node = std::make_shared<LinearNode<T>>();
            node->inputs.push_back(&input);
            node->inputs.push_back(&weight_);
            node->inputs.push_back(&bias_);
            output.grad_fn_ = node;
        }
        return output;
    }

    void zero_grad() {
        if (requires_grad_params_) {
            weight_.zero_grad();
            bias_.zero_grad();
        }
    }

    void update_weights(T learning_rate) {
        if (requires_grad_params_) {
            for (size_t i = 0; i < weight_.numel(); ++i) {
                weight_.data_[i] -= learning_rate * weight_.grad_data()[i];
            }
            for (size_t i = 0; i < bias_.numel(); ++i) {
                bias_.data_[i] -= learning_rate * bias_.grad_data()[i];
            }
        }
    }
};

template <typename T> class MSENode : public AutogradNode<T> {
  public:
    // inputs: [0] = predictions, [1] = targets
    void backward(const std::vector<T>& grad_from_downstream) override {
        // L_MSE = (1/N) * sum((pred_i - target_i)^2)
        // dL_MSE / dPred_i = (2/N) * (pred_i - target_i)
        // get dL_final / dL_MSE (1 if MSE is the final loss)
        // dL_final / dPred_i = (dL_final / dL_MSE) * (dL_MSE / dPred_i)
        // or, (dL_final / dL_MSE) * (2/N) * (pred_i - target_i)
        // where we define, scale = (dL_final / dL_MSE) * (2/N)
        if (this->inputs.size() < 2)
            return;

        Tensor<T>* predictions = this->inputs[0];
        Tensor<T>* targets = this->inputs[1];

        if (predictions != nullptr && predictions->requires_grad_) {
            T dLfinal_dLmse = grad_from_downstream[0];
            T scale = dLfinal_dLmse * (T(2.0) / predictions->numel());

            std::vector<T> grad_for_predictions(predictions->numel());
            for (size_t i = 0; i < predictions->numel(); ++i) {
                grad_for_predictions[i] =
                    scale * (predictions->data_[i] - targets->data_[i]);
                predictions->grad_[i] += grad_for_predictions[i];
            }

            if (predictions->grad_fn_) {
                predictions->grad_fn_->backward(grad_for_predictions);
            }
        }
    }
};

template <typename T>
Tensor<T> mse_loss(Tensor<T>& predictions, Tensor<T>& targets) {
    assert(predictions.shape() == targets.shape() &&
           "Shapes of predictions and targets must match for MSE loss.");

    bool result_requires_grad = predictions.requires_grad(); // usually true
    // output of mse_loss is scalar
    Tensor<T> loss_tensor({1}, result_requires_grad);
    T total_squared_error = 0;

    for (size_t i = 0; i < predictions.numel(); ++i) {
        T diff = predictions.data_[i] - targets.data_[i];
        total_squared_error += diff * diff;
    }
    loss_tensor.data_[0] = total_squared_error / predictions.numel();

    if (result_requires_grad) {
        auto node = std::make_shared<MSENode<T>>();
        node->inputs.push_back(&predictions);
        node->inputs.push_back(&targets); // usually no grad
        loss_tensor.grad_fn_ = node;
    }
    return loss_tensor;
}
