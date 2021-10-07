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

#include "omp.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <glog/logging.h>
#include <include/ocr_det.h>
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");
// detection related
DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(det_model_dir, "", "Path of det inference model.");
DEFINE_int32(max_side_len, 960, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");
// classification related
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
// recognition related
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_string(char_list_file, "../../ppocr/utils/ppocr_keys_v1.txt", "Path of dictionary.");


using namespace std;
using namespace cv;
using namespace PaddleOCR;


bool pathExists(const std::string& path){
  struct stat statbuf;
  return stat(path.c_str(), &statbuf) == 0;
}


int main_det(std::vector<cv::String> cv_all_img_names, const YAML::Node& cfg) {
    std::vector<double> time_info = {0, 0, 0};
    DBDetector det(cfg["det"]);
    
    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<std::vector<std::vector<int>>> boxes;
      std::vector<double> det_times;

      det.Run(srcimg, boxes, &det_times);
  
      time_info[0] += det_times[0];
      time_info[1] += det_times[1];
      time_info[2] += det_times[2];
    }
    
    return 0;
}


int main_rec(std::vector<cv::String> cv_all_img_names, const YAML::Node& cfg) {
    std::vector<double> time_info = {0, 0, 0};
    CRNNRecognizer rec(cfg["rec"]);

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }

      std::vector<double> rec_times;
      rec.Run(srcimg, &rec_times);
        
      time_info[0] += rec_times[0];
      time_info[1] += rec_times[1];
      time_info[2] += rec_times[2];
    }
    
    return 0;
}


int main_system(std::vector<cv::String> cv_all_img_names, const YAML::Node& cfg) {
    DBDetector det(cfg["det"]);

    Classifier *cls = nullptr;
    if (FLAGS_use_angle_cls) {
      cls = new Classifier(cfg["cls"]);
    }

    CRNNRecognizer rec(cfg["rec"]);

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < cv_all_img_names.size(); ++i) {
      LOG(INFO) << "The predict img: " << cv_all_img_names[i];

      cv::Mat srcimg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << endl;
        exit(1);
      }
      std::vector<std::vector<std::vector<int>>> boxes;
      std::vector<double> det_times;
      std::vector<double> rec_times;
        
      det.Run(srcimg, boxes, &det_times);
    
      cv::Mat crop_img;
      for (int j = 0; j < boxes.size(); j++) {
        crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);

        if (cls != nullptr) {
          crop_img = cls->Run(crop_img);
        }
        rec.Run(crop_img, &rec_times);
      }
        
      auto end = std::chrono::system_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      std::cout << "Cost  "
                << double(duration.count()) *
                       std::chrono::microseconds::period::num /
                       std::chrono::microseconds::period::den
                << "s" << std::endl;
    }
      
    return 0;
}


int main(int argc, char **argv) {
  // ================================================================
  // parsing args
  if(argc!=2){
    std::cerr << "Please run command: ./build/ppocr path/to/config.yaml" << std::endl;
    return -1;
  }else if(strcmp(argv[1], "-h")==0 || strcmp(argv[1], "--help")==0){
    std::cerr << "Please run command: ./build/ppocr path/to/config.yaml" << std::endl;
    return -1;
  }else if(!pathExists(argv[1])){
    std::cerr << "No such file: " << argv[1] << std::endl;
    std::cerr << "Please run command: ./build/ppocr path/to/config.yaml" << std::endl;
    return -1;
  }

  // ================================================================
  // load yaml config
  YAML::Node cfg = YAML::LoadFile(argv[1]);
  std::string mode = cfg["mode"].as<std::string>();
  std::string image_dir = cfg["image_dir"].as<std::string>();
        
  // ================================================================
  // check image_dir and colect images
  if (!pathExists(image_dir)) {
      std::cerr << "[ERROR] image path not exist! image_dir: " << image_dir << endl;
      exit(1);      
  }
  std::vector<cv::String> cv_all_img_names;
  cv::glob(image_dir, cv_all_img_names);
  std::cout << "total images num: " << cv_all_img_names.size() << endl;

  // ================================================================
  // do inference
  if (mode=="det") {
      return main_det(cv_all_img_names, cfg["models"]);
  }
  if (mode=="rec") {
      return main_rec(cv_all_img_names, cfg["models"]);
  }    
  if (mode=="system") {
      return main_system(cv_all_img_names, cfg["models"]);
  } 

}
