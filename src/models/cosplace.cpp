#include "models/cosplace.h"

#include <utility>

namespace viewfeat {

CosPlace::CosPlace(CosPlaceConfig cosplace_config) : descriptor_(std::move(cosplace_config), "CosPlace") {}

bool CosPlace::Build() { return descriptor_.Build(); }

bool CosPlace::Infer(const cv::Mat &image, Eigen::VectorXf &descriptor) { return descriptor_.Infer(image, descriptor); }

}  // namespace viewfeat
