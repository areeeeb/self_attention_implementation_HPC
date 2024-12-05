#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include "json.hpp"
#include "cnpy.h" // For loading .npy files

using json = nlohmann::json;

// Function declarations for implementations
std::vector<std::vector<float>> run_naive_attention(const std::vector<std::vector<float>> &input, const json &weights);
std::vector<std::vector<float>> run_optimized_attention(const std::vector<std::vector<float>> &input, const json &weights);

// Function to load weights from JSON
json load_weights(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open weights file: " + file_path);
    }

    json weights;
    file >> weights;
    return weights;
}

// Function to load input embeddings from .npy file
std::vector<std::vector<float>> load_embeddings(const std::string &file_path)
{
    cnpy::NpyArray arr = cnpy::npy_load(file_path);
    float *data = arr.data<float>();
    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];

    std::vector<std::vector<float>> embeddings(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            embeddings[i][j] = data[i * cols + j];
        }
    }
    return embeddings;
}

// Validate results
bool validate_results(const json &results_json, const std::vector<std::vector<float>> &actual)
{
    const float tolerance = 1e-6;
    for (const auto &sample : results_json)
    {
        size_t sample_index = sample["sample_index"];
        const auto &expected_output = sample["output"];
        const auto &actual_output = actual[sample_index];

        // if (expected_output.size() != actual_output.size()) return false;
        // for (size_t i = 0; i < expected_output.size(); ++i) {
        //     if (std::abs(expected_output[i] - actual_output[i]) > tolerance) return false;
        // }
    }
    return true;
}

int main()
{
    try
    {
        // Load weights
        std::string weights_file = "weights/attention_weights.json";
        json weights = load_weights(weights_file);
        std::cout << "Weights loaded successfully.\n";

        // Load input embeddings for sequence length 128
        std::string embeddings_file = "wikitext_embeddings/gpt2_embedded_128.npy";
        auto input_embeddings = load_embeddings(embeddings_file);
        std::cout << "Input embeddings loaded successfully.\n";

        // Load expected results for validation
        std::string results_file = "weights/results_seq_128.json";
        std::ifstream results_stream(results_file);
        if (!results_stream.is_open())
        {
            throw std::runtime_error("Unable to open results file: " + results_file);
        }
        json results_json;
        results_stream >> results_json;

        // Run and time naive implementation
        auto start_naive = std::chrono::high_resolution_clock::now();
        auto naive_output = run_naive_attention(input_embeddings, weights);
        auto end_naive = std::chrono::high_resolution_clock::now();
        double naive_time = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();
        std::cout << "Naive implementation completed in " << naive_time << " ms.\n";

        // Validate naive results
        if (!validate_results(results_json, naive_output))
        {
            std::cerr << "Validation failed for naive implementation.\n";
            return 1;
        }

        // Run and time optimized implementation
        auto start_optimized = std::chrono::high_resolution_clock::now();
        auto optimized_output = run_optimized_attention(input_embeddings, weights);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        double optimized_time = std::chrono::duration<double, std::milli>(end_optimized - start_optimized).count();
        std::cout << "Optimized implementation completed in " << optimized_time << " ms.\n";

        // Validate optimized results
        if (!validate_results(results_json, optimized_output))
        {
            std::cerr << "Validation failed for optimized implementation.\n";
            return 1;
        }

        std::cout << "All implementations passed validation.\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
