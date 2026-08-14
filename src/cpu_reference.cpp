#include "attention.h"
#include <cmath>
#include <cstdlib>
#include <cfloat>

// CPU reference: standard scaled dot-product attention
// This is the "ground truth" we validate GPU kernels against.
void attention_forward_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int seq_len,
    int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // For each query row i
    for (int i = 0; i < seq_len; i++) {
        // Step 1: Compute scores = Q[i] · K[j]^T for all j
        float* scores = (float*)malloc(seq_len * sizeof(float));
        float max_score = -FLT_MAX;

        for (int j = 0; j < seq_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q[i * head_dim + d] * K[j * head_dim + d];
            }
            scores[j] = dot * scale;
            if (scores[j] > max_score) {
                max_score = scores[j];
            }
        }

        // Step 2: Softmax (numerically stable: subtract max)
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j < seq_len; j++) {
            scores[j] /= sum_exp;
        }

        // Step 3: Output = scores × V
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                val += scores[j] * V[j * head_dim + d];
            }
            O[i * head_dim + d] = val;
        }

        free(scores);
    }
}
