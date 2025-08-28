#include "performance_evaluator.h"
#include <algorithm>
#include <cmath>

PerformanceEvaluator::PerformanceEvaluator(const EvaluationConfig& config)
    : config_(config) {
}

std::tuple<float, float, float, float, float> PerformanceEvaluator::evaluate(
    const std::vector<InferenceResult>& results) {
    
    std::vector<float> scores;
    scores.reserve(results.size());
    
    // 按sample_id排序以确保顺序正确
    auto sorted_results = results;
    std::sort(sorted_results.begin(), sorted_results.end(),
              [](const InferenceResult& a, const InferenceResult& b) {
                  return a.sample_id < b.sample_id;
              });
    
    for (const auto& result : sorted_results) {
        scores.push_back(result.score);
    }
    
    int total_samples = config_.n_positive + config_.n_negative;
    auto labels = _generate_labels(total_samples);
    auto predictions = _make_predictions(scores);
    
    // 计算TP, FP
    int n_tp = 0, n_fp = 0;
    for (int i = 0; i < total_samples; ++i) {
        if (predictions[i] == 1 && labels[i] == 1) n_tp++;
        if (predictions[i] == 1 && labels[i] == 0) n_fp++;
    }
    
    float tpr = static_cast<float>(n_tp) / config_.n_positive;
    float fpr = static_cast<float>(n_fp) / config_.n_negative;
    float ba = (tpr + (1 - fpr)) / 2;
    float acc = static_cast<float>(n_tp + (config_.n_negative - n_fp)) / total_samples;
    float auc = _calculate_auc(scores, labels);
    
    return {ba, acc, tpr, fpr, auc};
}

std::vector<int> PerformanceEvaluator::_generate_labels(int total_samples) {
    std::vector<int> labels(total_samples, 1);
    std::fill(labels.begin() + config_.n_positive, labels.end(), 0);
    return labels;
}

std::vector<int> PerformanceEvaluator::_make_predictions(const std::vector<float>& scores) {
    std::vector<int> predictions;
    predictions.reserve(scores.size());
    
    for (float score : scores) {
        predictions.push_back(score >= config_.threshold ? 1 : 0);
    }
    
    return predictions;
}

float PerformanceEvaluator::_calculate_auc(const std::vector<float>& scores, 
                                          const std::vector<int>& labels) {
    std::vector<float> fpr_list, tpr_list;
    
    for (float threshold = 0.0; threshold <= 1.0; threshold += 0.01) {
        int tp = 0, fp = 0;
        for (size_t i = 0; i < scores.size(); ++i) {
            if (scores[i] >= threshold && labels[i] == 1) tp++;
            if (scores[i] >= threshold && labels[i] == 0) fp++;
        }
        fpr_list.push_back(static_cast<float>(fp) / config_.n_negative);
        tpr_list.push_back(static_cast<float>(tp) / config_.n_positive);
    }
    
    float auc = 0.0;
    for (size_t i = 1; i < fpr_list.size(); ++i) {
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2;
    }
    
    return std::abs(auc);
}
