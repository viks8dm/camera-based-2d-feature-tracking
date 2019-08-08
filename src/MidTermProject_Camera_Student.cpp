/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include <stdlib.h>
#include <iterator>

using namespace std;

// Parameters to set before Build

string detectorType = "ORB"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
string descriptorType = "FREAK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
string buffDescriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

// task only used for saving data to file
int task = 7;
bool save_data = false;
// 7 -- find number of keypoints for each detectorType
// 8 - count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors
// 9 -- log the time it takes for keypoint detection and descriptor extraction


/***************************************************************/
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    std::vector< std::vector<float>> neighborhood_size;

    /* MAIN LOOP OVER ALL IMAGES */
    // define arrays to collect data for final prints
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        // compare size of dataBuffer with the size-limit
        if (dataBuffer.size() >= dataBufferSize) {
          dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
        cout << "image number: " << imgIndex << endl;

        /* DETECT IMAGE KEYPOINTS */
        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double time_detect = 0.0;

        if (detectorType.compare("SHITOMASI") == 0) {
          time_detect = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0) {
          time_detect = detKeypointsHarris(keypoints, imgGray, false);
        }
        else { // for all other key-point-detectors
          time_detect = detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        //// EOF STUDENT ASSIGNMENT

        // for debugging
        if  (false)
        {
            string windowName = "keypoints";
            cv::drawKeypoints(img, keypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::namedWindow(windowName, 6);
            imshow(windowName, img);
            cv::waitKey(0);
        }

        // write # of keypoints data to file
        fstream file_keypoints;
        if (task==7 && save_data) {
          file_keypoints.open("../results/" + detectorType + "_num_keypoints.txt", ios::app);
          file_keypoints << keypoints.size();
        }

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        int kp_count = 0;
        std::vector<float> temp_size;
        if (bFocusOnVehicle) { // verify if keypoint is within bounds
          for (auto it=keypoints.begin(); it<keypoints.end(); it++) {
            bool pointInRect = vehicleRect.contains((*it).pt); //
            if (!(pointInRect)) {
              keypoints.erase(it);
            }
            else {
              // get neighborhood size
              temp_size.push_back(it->size);
              kp_count++;
            }
          }
        }
        neighborhood_size.push_back(temp_size);

        // write # of vehicle keypoints data to file
        if (task==7 && save_data) {
          file_keypoints << '\t' << keypoints.size() << '\n';
          file_keypoints.close();
          continue;
        }

        //// EOF STUDENT ASSIGNMENT

        // check descriptor / detector / matcher combinations for compatibility
        if (descriptorType.compare("AKAZE") == 0) {
          if (detectorType.compare("AKAZE") != 0) {
            std::cout << "descriptorType: " << detectorType << " will NOT work with detectorType: AKAZE."<< endl;
            std::cout << "ERROR: AKAZE descriptor works only with AKAZE detector....exitting" << std::endl;
            exit(EXIT_FAILURE);
          }
        }

        if (descriptorType.compare("SIFT") == 0) {
          if (matcherType.compare("MAT_BF") == 0) {
            std::cout << "ERROR: SIFT is FLANN based....change matcherType to MAT_FLANN....exitting" << std::endl;
            exit(EXIT_FAILURE);
          }
        }

        if ((detectorType.compare("SIFT") == 0) && (descriptorType.compare("ORB") == 0)) {
            std::cout << "ERROR: SIFT detector and ORB descriptor do not work together....exitting" << std::endl;
            exit(EXIT_FAILURE);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        // cout << "#2 : DETECT KEYPOINTS done" << endl;


        /* EXTRACT KEYPOINT DESCRIPTORS */
        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        // string descriptorType = "BRIEF"; // BRIEF, ORB, FREAK, AKAZE, SIFT
        double time_extract = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, detectorType);
        if (task==9 && save_data) {
          fstream file_extract;
          file_extract.open("../results/" + detectorType + "_" + descriptorType  + "_dectect_extract_time.txt", ios::app);
          file_extract << time_detect << '\t' << time_extract << '\t' << (time_detect + time_extract) << '\n';
          file_extract.close();
        }

        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            double time_match = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, buffDescriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // write # of matches data to file
            if (task==8 && save_data) {
              fstream file_matches;
              file_matches.open("../results/" + detectorType + "_" + descriptorType + "_" + selectorType + "_matches.txt", ios::app);
              if (imgIndex==1) {
                file_matches << "prev-img-ID" << '\t' << "curr-img-ID" << '\t' << "num_matches" << '\t' << "time-match" << '\n';
                file_matches << "-----------" << '\t' << "-----------" << '\t' << "-----------" << '\t' << "-----------" << '\n';
              }
              file_matches << imgIndex-1 << '\t' << imgIndex << '\t' << matches.size() << '\t' << time_match << '\n';
              file_matches.close();
            }

            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

        cout << "=============================================" << endl;

    } // eof loop over all images

    if (task==7 && save_data) {
      ofstream file_neighbor_size("../results/" + detectorType  + "_kp_neighborhood_sizes.txt");
      ostream_iterator<int> output_iterator(file_neighbor_size, "\t");
      for ( int i = 0 ; i < neighborhood_size.size() ; i++ ) {
        copy(neighborhood_size.at(i).begin(), neighborhood_size.at(i).end(), output_iterator);
        file_neighbor_size << "\n";
      }
    }

    return 0;
}
