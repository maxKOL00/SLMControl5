#pragma once
#pragma once
// STL

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <chrono>

#include "Parameters.h"
#include "Display.h"
#include "CGHAlgorithm.cuh"
#include "ImageProcessing.cuh"

#include "ImageCapture.h"
#include "basic_fileIO.h"
#include "statusBox.h"

class mainThread {
	public:
		mainThread::mainThread(statusBox *box);
		void setDevice();
		int run_thread(std::string config);
		void camera_feedback_loop(const Parameters& params, std::unique_ptr<ImageCapture>& ic_ptr,
								  TweezerArray& tweezer_array, CGHAlgorithm& cgh, const ImageProcessing& ip,
								  byte* slm_image_ptr,
								  byte* phasemap_ptr,
								  const byte* phase_correction_ptr,
								  const byte* lut_ptr,
								  unsigned int number_of_pixels_unpadded);
		void correct_image(
			const ImageProcessing& ip,
			byte* slm_image_ptr, const byte* phasemap_ptr, const byte* phase_correction_ptr, const byte* lut_ptr
		);
	private:
	statusBox *editM;

};
