#ifndef VIEWFEAT_RETRIEVER_H_
#define VIEWFEAT_RETRIEVER_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "viewfeat/types.h"

namespace viewfeat {

// Global place descriptor. "default" and "cosplace" create CosPlace.
// "eigenplaces" creates EigenPlaces.
// Search paths are process-wide. SetSearchPaths is declared in types.h.
class GlobalRetriever {
 public:
  static std::unique_ptr<GlobalRetriever> Create(const std::string &family = "cosplace");

  GlobalRetriever(GlobalRetriever &&) noexcept;
  GlobalRetriever &operator=(GlobalRetriever &&) noexcept;
  ~GlobalRetriever();

  GlobalRetriever(const GlobalRetriever &) = delete;
  GlobalRetriever &operator=(const GlobalRetriever &) = delete;

  bool Infer(const cv::Mat &image, std::vector<float> &descriptor);
  int DescriptorDim() const;
  const std::string &Family() const;

 private:
  struct Impl;
  explicit GlobalRetriever(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace viewfeat

#endif  // VIEWFEAT_RETRIEVER_H_
