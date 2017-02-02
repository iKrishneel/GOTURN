// This file was mostly taken from the example given here:
// http://www.votchallenge.net/howto/integration.html

// Uncomment line below if you want to use rectangles
#define VOT_RECTANGLE
#include <string>

#include "native/vot.h"

#include "tracker/tracker.h"
#include "network/regressor_train.h"

std::string intToString(const int value, const int lenght = 0) {
    std::string s = std::to_string(value);
    int append = 7;
    if (value < 10) {
       append = 7;
    } else if (value > 9 && value < 100) {
       append = 6;
    } else if (value > 99 && value < 1000) {
       append = 5;
    }

    s = std::string(append, '0').append(s);
    return s;
    
}


int main(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " [gpu_id]" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  const string& model_file   = argv[1];
  const string& trained_file = argv[2];
  
  int gpu_id = 0;
  if (argc >= 4) {
    gpu_id = atoi(argv[3]);
  }

  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  // Ensuring randomness for fairness.
  srandom(time(NULL));

  // // Create a tracker object.
  const bool show_intermediate_output = false;
  Tracker tracker(show_intermediate_output);


  std::string extn = ".jpg";
  std::string vot_folder = "/home/krishneel/Documents/datasets/vot/vot2014/";
  std::string object_name = "bolt/";
  double box[8] = {
     324.29,220.10,346.51,162.22,371.71,171.90,349.49,229.78
  };

  int start_num = 1;
  int end_num = 1000;
  
  std::string path = vot_folder + object_name + intToString(start_num) + extn;
  cv::Mat image = cv::imread(path);


  BoundingBox region;
  region.x1_ = box[0];
  region.y1_ = box[3];
  region.x2_ = box[6];
  region.y2_ = box[7];
  
  cv::rectangle(image, cv::Point(box[0], box[3]), cv::Point(box[6], box[7]),
                cv::Scalar(0, 255, 0), 2);


  region.Print();
  region.DrawBoundingBox(&image);
  cv::imshow("image", image);
  cv::waitKey(0);

  tracker.Init(image, region, &regressor);

  
  while (start_num++ < end_num) {

      path = vot_folder + object_name + intToString(start_num) + extn;
      if (path.empty())
         break;
      
      image = cv::imread(path);
      if (image.empty())
         break;
      
      // Track and estimate the bounding box location.
      BoundingBox bbox_estimate;
      tracker.Track(image, &regressor, &bbox_estimate);

      bbox_estimate.Print();
      bbox_estimate.DrawBoundingBox(&image);

      std::cout << "------------"  << "\n";

      cv::imshow("image", image);
      cv::waitKey(0);
      
      
      // bbox_estimate.GetRegion(&region);
      // vot.report(region); // Report the position of the tracker
  }

  // Finishing the communication is completed automatically with the destruction
  // of the communication object (if you are using pointers you have to explicitly
  // delete the object).

  return 0;
}
