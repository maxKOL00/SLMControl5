#pragma once
#pragma once
// STL

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <chrono>

#include "Constants.h"
#include "Parameters.h"
#include "Display.h"
#include "CGHAlgorithm.cuh"
#include "ImageProcessing.cuh"
#include "statusBox.h"

class mainThread {
	public:
		mainThread::mainThread();
		void setDevice(statusBox &editM);
		int run_thread(statusBox &editM);
		void correct_image(
			const ImageProcessing& ip,
			byte* slm_image_ptr, const byte* phasemap_ptr, const byte* phase_correction_ptr, const byte* lut_ptr
		);
	private:

};
