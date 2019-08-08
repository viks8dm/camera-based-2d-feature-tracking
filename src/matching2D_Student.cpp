#include <numeric>
#include "matching2D.hpp"
#include <stdlib.h>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
double matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{

    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t = 0.0;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
      if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            // descRef.convertTo(descRef, CV_32F);
        }
      if (descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descRef.convertTo(descRef, CV_32F);
        }
      matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }


    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
      t = (double)cv::getTickCount();

      matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1

      t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
      cout << "SEL_NN with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
      int k = 2;
      double distRatio = 0.8;

      vector<vector<cv::DMatch>> knn_matches;
      t = (double)cv::getTickCount();
      matcher->knnMatch(descSource, descRef, knn_matches, k);

      // distance ratio filtering
      for (int i = 0; i < kPtsSource.size(); ++i) {
          if (knn_matches[i][0].distance < distRatio * knn_matches[i][1].distance) {
              matches.push_back(knn_matches[i][0]);
          }
      }
      t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
      cout << "SEL_KNN with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }

    return (1000 * t / 1.0);

}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, string detectorType)
{
    // select appropriate descriptor
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::DescriptorExtractor> extractor;

    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0) {
        /*
        https://docs.opencv.org/3.4/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html#ae3bc52666010fb137ab6f0d32de51f60
        Parameters:
        bytes	legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
        use_orientation	sample patterns using keypoints orientation, disabled by default.
        */
        int 	bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create	(bytes, use_orientation);
    }
    else if (descriptorType.compare("ORB") == 0) {
      /*
      https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
      */
      int   nfeatures = 500;     // The maximum number of features to retain.
      float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
      int   nlevels = 8;         // The number of pyramid levels.
      int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
      int   firstLevel = 0;      // The level of pyramid to put source image to.
      int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
      auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE -- to rank features.
      int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
      int   fastThreshold = 20;  // The FAST threshold.

      extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                                firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0) {
      /*
      https://docs.opencv.org/3.4.3/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html
      */
      bool	orientationNormalized = true;  // Enable orientation normalization.
      bool	scaleNormalized = true;        // Enable scale normalization.
      float patternScale = 22.0f;         // Scaling of the description pattern.
      int	nOctaves = 4;                     // Number of octaves covered by the detected keypoints.
      const std::vector< int > & 	selectedPairs = std::vector< int >(); // (Optional) user defined selected pairs indexes,

      extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale,
                                               nOctaves, selectedPairs);
    }
    else if (descriptorType.compare("AKAZE") == 0) {
      /*
      https://docs.opencv.org/3.4/d8/d30/classcv_1_1AKAZE.html
      */
      auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
      // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
      int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
      int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
      float threshold = 0.001f;         //   Detector response threshold to accept point.
      int   nOctaves = 4;               // Maximum octave evolution of the image.
      int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
      auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
      //                   DIFF_WEICKERT or DIFF_CHARBONNIER.

      extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                    threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if (descriptorType.compare("SIFT") == 0) {
      std::cout << "STARTING SIFT extractor" << std::endl;
      /*
      https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
      */

      int nfeatures = 0; // The number of best features to retain.
      int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
      // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
      double contrastThreshold = 0.04;
      double edgeThreshold = 10; // The threshold used to filter out edge-like features.
      double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave #0.

      extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
      // extractor = cv::xfeatures2d::SIFT::create();
    }
    else {
      std::cout << "ERROR in descriptor-type within descKeypoints() ....exitting" << std::endl;
      exit(EXIT_FAILURE);
    }

    // perform feature description
    // double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;

    // // write time for keypoint-extraction
    // fstream file_descKeypoints;
    // file_descKeypoints.open("../results/" + descriptorType + "_descriptor_" + detectorType + "_kpt_extract_time.txt", ios::app);
    // file_descKeypoints << 1000 * t / 1.0 << '\t' << '\n';
    // file_descKeypoints.close();

    return (1000 * t / 1.0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return (1000 * t / 1.0);
}


/**********************************************************************/
// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
  int blockSize = 2;  // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3; // aperture parameter for Sobel operator
  int minResponse = 100; // minimum value for a corner in the 8 bit scaled response matrix
  double k = 0.04;  //Haris parameter
  double t = (double)cv::getTickCount();
  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img,dst,blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  double maxOverlap = 0.0;
  for (int j = 0; j < dst_norm.rows; j++) {
      for (int i = 0; i < dst_norm.cols; i++) {
          int response = (int)dst_norm.at<float>(j, i);
          if (response > minResponse) {
              cv::KeyPoint newKeyPoint;
              newKeyPoint.pt = cv::Point2f(i, j);
              newKeyPoint.size = 2*apertureSize;
              newKeyPoint.response = response;
              keypoints.push_back(newKeyPoint);
          }
      }
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  if  (bVis)
  {
      string windowName = "Harris Corner Detection Results";
      cv::Mat visImage = dst_norm_scaled.clone();
      cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::namedWindow(windowName, 6);
      imshow(windowName, visImage);
      cv::waitKey(0);
  }


  return (1000 * t / 1.0);
}

/**********************************************************************/
// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {

  cv::Ptr<cv::FeatureDetector> detector;
  double t = (double)cv::getTickCount();

  if (detectorType == "FAST")
  {
    int threshold = 30;    // difference between intensity of the central pixel and pixels of a circle around this pixel
    bool bNMS = true;      // perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
    detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
  }
  else if (detectorType == "BRISK")
  {
    int threshold = 30;        //   AGAST detection threshold score
    int octaves = 3;           // detection octaves
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint
    detector = cv::BRISK::create(threshold, octaves, patternScale);
  }
  else if (detectorType == "ORB")
  {
    int   nfeatures = 500;     // The maximum number of features to retain.
    float scaleFactor = 1.2f;  // Pyramid decimation ratio, greater than 1.
    int   nlevels = 8;         // The number of pyramid levels.
    int   edgeThreshold = 31;  // This is size of the border where the features are not detected.
    int   firstLevel = 0;      // The level of pyramid to put source image to.
    int   WTA_K = 2;           // The number of points that produce each element of the oriented BRIEF descriptor.
    auto  scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE / FAST_SCORE algorithm is used to rank features.
    int   patchSize = 31;      // Size of the patch used by the oriented BRIEF descriptor.
    int   fastThreshold = 20;  // The FAST threshold.
    detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold,
                               firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  }
  else if (detectorType == "AKAZE")
  {
    // Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT,
    //                                   DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
    auto  descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
    int   descriptor_size = 0;        // Size of the descriptor in bits. 0 -> Full size
    int   descriptor_channels = 3;    // Number of channels in the descriptor (1, 2, 3).
    float threshold = 0.001f;         //   Detector response threshold to accept point.
    int   nOctaves = 4;               // Maximum octave evolution of the image.
    int   nOctaveLayers = 4;          // Default number of sublevels per scale level.
    auto  diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity type. DIFF_PM_G1, DIFF_PM_G2,
    //                   DIFF_WEICKERT or DIFF_CHARBONNIER.
    detector = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                 threshold, nOctaves, nOctaveLayers, diffusivity);
  }
  else if (detectorType == "SIFT")
  {
    std::cout << "STARTING SIFT detector" << std::endl;

    int nfeatures = 0; // The number of best features to retain.
    int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
    // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
    double contrastThreshold = 0.04;
    double edgeThreshold = 10; // The threshold used to filter out edge-like features.
    double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave \#0.

    detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    // detector = cv::xfeatures2d::SIFT::create();
  }
  else
  {
    cout << "ERROR in detKeypointsModern() ....exitting" << endl;
    exit(EXIT_FAILURE);
  }

  // plot
  if (bVis)
  {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = detectorType + " Keypoint Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }

  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  return (1000 * t / 1.0);
}
