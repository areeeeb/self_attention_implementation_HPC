#include <vector>
#include <cmath>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

// Naive matrix multiplication: C = A * B
std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A, const std::vector<std::vector<float>> &B)
{
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner_dim = B.size();

    std::vector<std::vector<float>> C(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            for (size_t k = 0; k < inner_dim; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Add bias to matrix rows
void add_bias(std::vector<std::vector<float>> &matrix, const std::vector<float> &bias)
{
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            matrix[i][j] += bias[j];
        }
    }
}

// Softmax over rows of a matrix
void softmax(std::vector<std::vector<float>> &matrix)
{
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        float max_val = *std::max_element(matrix[i].begin(), matrix[i].end());
        float sum = 0.0f;

        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            matrix[i][j] = std::exp(matrix[i][j] - max_val); // Avoid overflow
            sum += matrix[i][j];
        }
        for (size_t j = 0; j < matrix[i].size(); ++j)
        {
            matrix[i][j] /= sum;
        }
    }
}

// Naive self-attention computation
std::vector<std::vector<float>> run_naive_attention(const std::vector<std::vector<float>> &input, const json &weights)
{
    size_t seq_len = input.size();
    size_t hidden_size = input[0].size();

    // Load weights and biases
    auto query_weight = weights["query_weight"].get<std::vector<std::vector<float>>>();
    auto key_weight = weights["key_weight"].get<std::vector<std::vector<float>>>();
    auto value_weight = weights["value_weight"].get<std::vector<std::vector<float>>>();
    auto output_weight = weights["output_weight"].get<std::vector<std::vector<float>>>();
    auto query_bias = weights["query_bias"].get<std::vector<float>>();
    auto key_bias = weights["key_bias"].get<std::vector<float>>();
    auto value_bias = weights["value_bias"].get<std::vector<float>>();
    auto output_bias = weights["output_bias"].get<std::vector<float>>();

    // Step 1: Compute Q, K, V
    auto Q = matmul(input, query_weight);
    add_bias(Q, query_bias);

    auto K = matmul(input, key_weight);
    add_bias(K, key_bias);

    auto V = matmul(input, value_weight);
    add_bias(V, value_bias);

    // Step 2: Compute attention scores
    auto scores = matmul(Q, K); // QK^T
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[i].size(); ++j)
        {
            scores[i][j] /= std::sqrt(hidden_size);
        }
    }
    softmax(scores);

    // Step 3: Compute attention output
    auto attention_output = matmul(scores, V);

    // Step 4: Final linear transformation
    auto output = matmul(attention_output, output_weight);
    add_bias(output, output_bias);

    return output;
}
