#include "models/eigenplaces.h"

#include <utility>

namespace viewfeat {

EigenPlaces::EigenPlaces(EigenPlacesConfig eigenplaces_config) : descriptor_(std::move(eigenplaces_config), "EigenPlaces") {}

bool EigenPlaces::Build() { return descriptor_.Build(); }

bool EigenPlaces::Infer(const cv::Mat &image, Eigen::VectorXf &descriptor) { return descriptor_.Infer(image, descriptor); }

}  // namespace viewfeat
