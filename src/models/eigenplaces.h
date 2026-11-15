#ifndef VIEWFEAT_MODELS_EIGENPLACES_H_
#define VIEWFEAT_MODELS_EIGENPLACES_H_

#include <Eigen/Core>

#include <opencv2/core/mat.hpp>

#include "models/place_descriptor.h"
#include "runtime/config.h"

namespace viewfeat {

class EigenPlaces {
 public:
  explicit EigenPlaces(EigenPlacesConfig eigenplaces_config);

  bool Build();

  // Run EigenPlaces on a BGR/grayscale/BGRA image and return an L2-normalized descriptor.
  bool Infer(const cv::Mat &image, Eigen::VectorXf &descriptor);

 private:
  PlaceDescriptor descriptor_;
};

}  // namespace viewfeat

#endif  // VIEWFEAT_MODELS_EIGENPLACES_H_
