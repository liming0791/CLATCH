/*******************************************************************
*   main.cpp
*   CLATCH
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Sep 12, 2016
*******************************************************************/
//
// Fastest implementation of the fully scale-
// and rotation-invariant LATCH 512-bit binary
// feature descriptor as described in the 2015
// paper by Levi and Hassner:
//
// "LATCH: Learned Arrangements of Three Patch Codes"
// http://arxiv.org/abs/1501.03719
//
// See also the ECCV 2016 Descriptor Workshop paper, of which I am a coauthor:
//
// "The CUDA LATCH Binary Descriptor"
// http://arxiv.org/abs/1609.03986
//
// And the original LATCH project's website:
// http://www.openu.ac.il/home/hassner/projects/LATCH/
//
// This implementation is insanely fast, matching or beating
// the much simpler ORB descriptor despite outputting twice
// as many bits AND being a superior descriptor.
//
// NOTE: angles are in radians!! You can change this
// behavior if it's a problem for your pipeline.
//
// A key insight responsible for much of the performance of
// this laboriously crafted CUDA kernel is due to
// Christopher Parker (https://github.com/csp256), to whom
// I am extremely grateful.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CLATCH.h
// and CLATCH.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include "Clatch.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

int main() {
	// ------------- Configuration ------------
	constexpr int numkps = 1000;
	constexpr char name[] = "test.jpg";
	// --------------------------------


	// ------------- Image Read ------------
	cv::Mat image = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cerr << "ERROR: failed to open image. Aborting." << std::endl;
		return EXIT_FAILURE;
	}
	// --------------------------------


	// ------------- Detection ------------
	std::cout << std::endl << "Detecting..." << std::endl;
	cv::Ptr<cv::ORB> orb = cv::ORB::create(numkps, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);
	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [image](const cv::KeyPoint& kp) {return kp.pt.x <= 36 || kp.pt.y <= 36 || kp.pt.x >= image.cols - 36 || kp.pt.y >= image.rows - 36; }), keypoints.end());
	// --------------------------------


	// ------------- CLATCH ------------
    std::cout << std::endl << "CLATCH..." << std::endl;
    cv::Mat desc;
    Clatch clatch;

    clatch.compute(image, keypoints, desc);
    clatch.compute(image, keypoints, desc);
    clatch.compute(image, keypoints, desc);

    return 0;
}
