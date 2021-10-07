// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>
#include <yaml-cpp/yaml.h>

#include <include/ocr_cls.h>
#include <include/postprocess_op.h>
#include <include/preprocess_op.h>
#include <include/utility.h>

using namespace paddle_infer;

namespace PaddleOCR {

class CRNNRecognizer {
public:
  explicit CRNNRecognizer(const YAML::Node& cfg) {
    this->cpu_math_library_num_threads_ = cfg["cpu_math_library_num_threads"].as<int>();
    this->use_mkldnn_ = cfg["use_mkldnn"].as<bool>();
    this->precision_ = cfg["precision"].as<std::string>();

    this->label_list_ = Utility::ReadDict(cfg["label_path"].as<std::string>());
    this->label_list_.insert(this->label_list_.begin(),
                             "#"); // blank char for ctc
    this->label_list_.push_back(" ");

    std::string model_dir = cfg["model_dir"].as<std::string>();
    LoadModel(model_dir);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir);

  void Run(cv::Mat &img, std::vector<double> *times);

private:
  std::shared_ptr<Predictor> predictor_;

  bool use_gpu_ = false;
  int gpu_id_ = 0;
  int gpu_mem_ = 4000;
  int cpu_math_library_num_threads_ = 4;
  bool use_mkldnn_ = false;

  std::vector<std::string> label_list_;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt_ = false;
  std::string precision_ = "fp32";
  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  Permute permute_op_;

  // post-process
  PostProcessor post_processor_;

}; // class CrnnRecognizer

} // namespace PaddleOCR
