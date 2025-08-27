#pragma once
#include <vector>
#include <tuple>
#include "data_types.h"

class PerformanceEvaluator {
private:
    EvaluationConfig config_;

public:
    explicit PerformanceEvaluator(const EvaluationConfig& config);
    
    std::tuple<float, float, float, float, float> evaluate(
        const std::vector<InferenceResult>& results
    );
    
private:
    std::vector<int> generate_labels(int total_samples);
    std::vector<int> make_predictions(const std::vector<float>& scores);
    float calculate_auc(const std::vector<float>& scores, const std::vector<int>& labels);
};
