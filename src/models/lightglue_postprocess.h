#ifndef VIEWFEAT_MODELS_LIGHTGLUE_POSTPROCESS_H_
#define VIEWFEAT_MODELS_LIGHTGLUE_POSTPROCESS_H_

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace viewfeat {

// Keypoints are pixel coordinates. LightGlue wants them normalized by the longer image side.
// Descriptors are copied from row 2 of each feature column. Padding stays with the caller.
inline void PackAndNormalizeFeatures(const float *feature_data, int num_cols, int feature_rows, int desc_dim, int width, int height,
                                     float *keypoints_buffer, float *descriptors_buffer) {
  const float norm_factor = 1.0F / (static_cast<float>(std::max(width, height)) * 0.5F);
  const float cx = static_cast<float>(width / 2);
  const float cy = static_cast<float>(height / 2);
  for (int col = 0; col < num_cols; ++col) {
    const float *col_data = feature_data + col * feature_rows;
    keypoints_buffer[col * 2] = (col_data[0] - cx) * norm_factor;
    keypoints_buffer[col * 2 + 1] = (col_data[1] - cy) * norm_factor;
    std::memcpy(descriptors_buffer + col * desc_dim, col_data + 2, static_cast<size_t>(desc_dim) * sizeof(float));
  }
}

// scores is (rows x cols), row-major, with the LightGlue dustbin in the last row and column.
// Mutual nearest neighbors are taken over the first rows-1 by cols-1 entries.
// valid_rows / valid_cols drop padded keypoints. Pass rows-1 and cols-1 when every keypoint is real.
inline void MutualNearestMatches(const float *scores, int rows, int cols, float log_threshold, int valid_rows, int valid_cols,
                                 Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score) {
  const int n0 = rows - 1;
  const int n1 = cols - 1;
  if (n0 <= 0 || n1 <= 0) {
    matches_index.resize(0, 2);
    matches_score.resize(0, 1);
    return;
  }

  std::vector<std::pair<int, float>> row_max(n0);
  for (int row = 0; row < n0; ++row) {
    const float *row_ptr = scores + row * cols;
    int best_col = 0;
    float best_val = row_ptr[0];
    for (int col = 1; col < n1; ++col) {
      if (row_ptr[col] > best_val) {
        best_val = row_ptr[col];
        best_col = col;
      }
    }
    row_max[row] = std::make_pair(best_col, best_val);
  }

  std::vector<std::pair<int, float>> col_max(n1);
  for (int col = 0; col < n1; ++col) {
    int best_row = 0;
    float best_val = scores[col];
    for (int row = 1; row < n0; ++row) {
      const float val = scores[row * cols + col];
      if (val > best_val) {
        best_val = val;
        best_row = row;
      }
    }
    col_max[col] = std::make_pair(best_row, best_val);
  }

  std::vector<int> matches_index0_vec;
  std::vector<int> matches_index1_vec;
  std::vector<float> matches_score_vec;
  for (int row = 0; row < n0; ++row) {
    if (row >= valid_rows || row_max[row].first >= valid_cols) {
      continue;
    }
    if (row == col_max[row_max[row].first].first && row_max[row].second > log_threshold) {
      matches_index0_vec.push_back(row);
      matches_index1_vec.push_back(row_max[row].first);
      matches_score_vec.push_back(std::exp(row_max[row].second));
    }
  }

  matches_index.resize(matches_index0_vec.size(), 2);
  matches_score.resize(matches_score_vec.size(), 1);
  for (int i = 0; i < static_cast<int>(matches_index0_vec.size()); ++i) {
    matches_index(i, 0) = matches_index0_vec[i];
    matches_index(i, 1) = matches_index1_vec[i];
    matches_score(i) = matches_score_vec[i];
  }
}

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_LIGHTGLUE_POSTPROCESS_H_
