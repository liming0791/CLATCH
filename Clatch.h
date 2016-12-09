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

#ifndef CLATCH_H
#define CLATCH_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "CLATCH.h"

using namespace std::chrono;

class Clatch
{
    public:
        Clatch();
        ~Clatch();
        void compute(const cv::Mat &img, std::vector<cv::KeyPoint> &kps, cv::Mat &desc);
    private:
        cudaTextureObject_t d_trip_tex;

};

#endif
