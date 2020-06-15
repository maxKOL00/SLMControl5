#pragma once
// STL
#include "mainThread.h"
#include "ImageCapture.h"
#include <sstream>


mainThread::mainThread() {
}

void mainThread::setDevice(statusBox &editM) {
    int count = 0;
    std::string GPU_name = "";
    cudaGetDeviceCount(&count);//expect two GPUs 
    cudaDeviceProp prop;
    for (int i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            throw std::runtime_error("Could Not Find GPU Properties");
        }
        GPU_name = (prop.name);
        if (GPU_name == "GeForce RTX 2080 SUPER") {
            cudaSetDevice(i);//find the GPU we bought
            editM.appendMessage("Connected to GeForce RTX 2080 SUPER");
            printf("Connected to GeForce RTX 2080 SUPER");
            break;
        }
    }
}

int mainThread::run_thread(statusBox& editM) {
    try {
        //editM->clear();
        const auto params = Parameters();
        // define parameters to check their availability early on
        // This could be prevented by checking in params and using
        // default values if nothing is given
        const size_t num_px_tot = SLM_PX_X * SLM_PX_Y;

        const size_t camera_px_x = params.get_camera_px_x();
        const size_t camera_px_y = params.get_camera_px_y();

        const size_t lut_patch_size_x = params.get_lut_patch_size_x_px();
        const size_t lut_patch_size_y = params.get_lut_patch_size_y_px();
        const size_t lut_patch_num_x = params.get_number_of_lut_patches_x();
        const size_t lut_patch_num_y = params.get_number_of_lut_patches_x();

        const size_t num_traps_x = params.get_num_traps_x();
        const size_t num_traps_y = params.get_num_traps_y();

        const double spacing_x_um = params.get_spacing_x_um();
        const double spacing_y_um = params.get_spacing_y_um();

        const double radial_shift_x_um = params.get_radial_shift_x_um();
        const double radial_shift_y_um = params.get_radial_shift_y_um();

        const double axial_shift_um = params.get_axial_shift_um();

        const double beam_waist_x_mm = params.get_beam_waist_x_mm();
        const double beam_waist_y_mm = params.get_beam_waist_y_mm();

        const std::string output_folder = params.get_output_folder();

        // Camera feedback related variables
        // The ImageCapture instance is only created if feedback is enabled.
        // The program catches all ImageCaptureExceptions and terminates
        // in a controlled way

        std::unique_ptr<ImageCapture> ic_ptr;
        const bool camera_feedback_enabled = params.get_camera_feedback_enabled();

        if (camera_feedback_enabled) {
            if (USE_GUI) {
                editM.appendMessage("Camera feedback enabled\n");
            }
            else {
                printf("Camera feedback enabled\n");
            }
            try {
                ic_ptr = std::make_unique<ImageCapture>(params);
            }
            catch (const ImageCaptureException& e) {
                std::cout << e.what() << "\n";

                // It seems that manually calling the params dtor
                // gives a faster cleanup
                params.~Parameters();
                //std::cout << "Press any key to close window . . ." << std::endl;
                //std::cin.get();
                return EXIT_FAILURE;
            }
        }

        const size_t max_iterations_camera_feedback = params.get_max_iterations_camera_feedback();
        const double max_nonuniformity_camera_feedback_percent = params.get_max_nonuniformity_camera_feedback_percent();
        const bool save_data = params.get_save_data();

        // Init before cuda stuff is allocated
        const auto ip = ImageProcessing(params);
        auto cgh = CGHAlgorithm(params, &editM);
        auto tweezer_array = TweezerArray(params);

        byte* slm_image_ptr;
        byte* phase_correction_ptr;
        byte* lut_ptr;
        byte* phasemap_ptr;

        setDevice(editM);//find the GPU
        // Allocate memory
        {
            if (cudaSuccess != cudaMallocManaged(&slm_image_ptr, num_px_tot * sizeof(byte))) {
                std::cerr << "Could not allocate memory for slm_image_ptr.\n";
                return EXIT_FAILURE;
            }
            if (cudaSuccess != cudaMallocManaged(&lut_ptr, 256 * lut_patch_num_x * lut_patch_num_y * sizeof(byte))) {
                std::cerr << "Could not allocate memory for lut_ptr.\n";
                return EXIT_FAILURE;
            }
            if (cudaSuccess != cudaMallocManaged(&phase_correction_ptr, num_px_tot * sizeof(byte))) {
                std::cerr << "Could not allocate memory for phase_correction_ptr.\n";
                return EXIT_FAILURE;
            }
            // Allocate square array
            if (cudaSuccess != cudaMallocManaged(&phasemap_ptr, SLM_PX_Y * SLM_PX_Y * sizeof(byte))) {
                std::cerr << "Could not allocate memory for phasemap_ptr.\n";
                return EXIT_FAILURE;
            }
            if (cudaSuccess != cudaDeviceSynchronize()) {
                std::cerr << "Could not synchronize.\n";
                return EXIT_FAILURE;
            }
        }

        basic_fileIO::load_LUT(lut_ptr, lut_patch_num_x, lut_patch_num_y);
        basic_fileIO::load_phase_correction(phase_correction_ptr, SLM_PX_X, SLM_PX_Y);

        cudaDeviceSynchronize();
        if (USE_GUI) {
            editM.appendMessage("Read correction files");
        }
        else {
             std::cout << "Read correction files\n";
        }

        const auto start = std::chrono::system_clock::now();

        const auto non_uniformity_vec = cgh.AWGS2D_loop(tweezer_array, phasemap_ptr);

        ip.shift_fourier_image(phasemap_ptr, radial_shift_x_um, radial_shift_y_um);
        ip.fresnel_lens(phasemap_ptr, SLM_PX_Y, SLM_PX_Y, axial_shift_um);

        correct_image(ip, slm_image_ptr, phasemap_ptr, phase_correction_ptr, lut_ptr);

        // Save output
        if (save_data) {

            cgh.save_output_intensity_distribution(
                output_folder + "theory_output.bmp"
            );

            cgh.save_input_phase_distribution(
                output_folder + "phasemap_before_correction.bmp"
            );

            basic_fileIO::save_one_column_data(
                output_folder + "nonuniformity.txt",
                non_uniformity_vec.cbegin(), non_uniformity_vec.cend()
            );

            //basic_fileIO::save_one_column_data(
            //    output_folder + "mean_intensity.txt",
            //    mean_intensity_vec.cbegin(), mean_intensity_vec.cend()
            //);
        }

        init_window(params);
        Sleep(500);

        // glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
        // glutCloseFunc(cleanup);
        size_t cnt = 0;
        while (cnt < 3) {
            display_phasemap(slm_image_ptr);
            Sleep(100);
            cnt++;
        }

        // Camera feedback loop
        if (camera_feedback_enabled) {
            if (USE_GUI) {
                editM.appendMessage("Starting camera feedback");
            }
            else {
                printf("\nStarting camera feedback\n\n");
            }

            if (ic_ptr) {
                ic_ptr->adjust_exposure_time_automatically(230, 10, &editM);
            }
            else {
                throw std::runtime_error("ic_ptr is null");
            }

            std::vector<byte> image_data(camera_px_x * camera_px_y);

            // Save before feedback
            if (save_data) {
                if (ic_ptr) {
                    ic_ptr->capture_image(image_data.data(), camera_px_x, camera_px_y);
                }
                else {
                    throw std::runtime_error("ic_ptr is null");
                }
                basic_fileIO::save_as_bmp(
                    output_folder + "before_feedback.bmp",
                    image_data.data(), camera_px_x, camera_px_y
                );
            }

            double delta = 100.0;

            size_t iteration = 0;

            // I know it's the same name as above but it's a different scope
            std::vector<double> non_uniformity_vec;

            while ((iteration < max_iterations_camera_feedback) && (100.0 * delta > max_nonuniformity_camera_feedback_percent)) {

                if (ic_ptr) {
                    ic_ptr->capture_image(image_data.data(), camera_px_x, camera_px_y);
                    if (save_data) {
                        std::stringstream ss;
                        ss << output_folder << iteration << ".bmp";
                        basic_fileIO::save_as_bmp(ss.str(), image_data.data(), camera_px_x, camera_px_y);
                    }
                }
                else {
                    throw std::runtime_error("ic_ptr is null");
                }
                // Undo the shift
                ip.shift_fourier_image(phasemap_ptr, -radial_shift_x_um, -radial_shift_y_um);
                ip.fresnel_lens(phasemap_ptr, SLM_PX_Y, SLM_PX_Y, -axial_shift_um);

                // The camera image is flipped about both axis so we have to undo that
                ip.invert_camera_image(image_data.data(), camera_px_x, camera_px_y);

                // Because the optical system is jittering quite a lot peak position can drift by up to a few px
                // between images. If the system is stabilized later this only needs to be done once in the beginning.
                const auto sorted_flattened_peak_indices = ip.create_mask(image_data.data(), camera_px_x, camera_px_y, num_traps_x, num_traps_y);

                tweezer_array.update_position_in_camera_image(sorted_flattened_peak_indices);

                delta = cgh.AWGS2D_camera_feedback_iteration(
                    tweezer_array,
                    image_data.data(),
                    phasemap_ptr
                );

                non_uniformity_vec.push_back(delta);

                if (USE_GUI) {
                    std::stringstream stream;
                    stream << std::setfill('0') << std::setw(long long(log10(max_iterations_camera_feedback) + 1));
                    editM.appendMessage(stream.str().c_str());
                    stream.str(std::string());
                    stream << iteration + 1 << "/" << max_iterations_camera_feedback << "; ";
                    editM.appendMessage(stream.str().c_str());
                    stream.str(std::string());
                    stream << "Non-uniformity: " << std::setw(3) << 100 * delta << "%\n";
                    editM.appendMessage(stream.str().c_str());
                }
                else {
                    std::cout << std::setfill('0') << std::setw(long long(log10(max_iterations_camera_feedback) + 1));
                    std::cout << iteration + 1 << "/" << max_iterations_camera_feedback << "; ";
                    std::cout << "Non-uniformity: " << std::setw(3) << 100 * delta << "%\n";
                }

                ip.shift_fourier_image(phasemap_ptr, radial_shift_x_um, radial_shift_y_um);
                ip.fresnel_lens(phasemap_ptr, SLM_PX_Y, SLM_PX_Y, axial_shift_um);

                correct_image(ip, slm_image_ptr, phasemap_ptr, phase_correction_ptr, lut_ptr);

                display_phasemap(slm_image_ptr);
                Sleep(200);
                iteration++;
            }

            // Save after feedback
            if (save_data) {
                if (ic_ptr) {
                    ic_ptr->capture_image(image_data.data(), camera_px_x, camera_px_y);
                }
                else {
                    throw std::runtime_error("ic_ptr is null");
                }
                basic_fileIO::save_as_bmp(
                    output_folder + "after_feedback.bmp",
                    image_data.data(), camera_px_x, camera_px_y
                );

                basic_fileIO::save_one_column_data(
                    output_folder + "non_uniformity_feedback.txt",
                    non_uniformity_vec.begin(), non_uniformity_vec.end()
                );

                cgh.save_input_phase_distribution(
                    output_folder + "phasemap_after_correction.bmp"
                );

                basic_fileIO::save_as_bmp(output_folder + "phasemap_total.bmp", slm_image_ptr, SLM_PX_X, SLM_PX_Y);
            }

            // Delete manually so that Vimba can be opened while the image is shown
            if (ic_ptr) {
                ic_ptr->~ImageCapture();
            }
            else {
                throw std::runtime_error("ic_ptr is null");
            }
        }

        const auto end = std::chrono::system_clock::now();
        const size_t diff_in_s = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        if (USE_GUI) {
            std::stringstream stream;
            stream << "Iterating took: " << (diff_in_s / 60) << "min " << (diff_in_s % 60) << "s ";
            editM.appendMessage(stream.str().c_str());
        }
        else {
            std::cout << "Iterating took: " << (diff_in_s / 60) << "min " << (diff_in_s % 60) << "s\n";
        }

        display_phasemap(slm_image_ptr);
        std::cout << "Showing image, press Enter key to quit\n";
        std::cin.get();

        cudaFree(slm_image_ptr);
        cudaFree(lut_ptr);
        cudaFree(phase_correction_ptr);
        cudaFree(phasemap_ptr);

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {

        std::cout << e.what() << "\n";
        //std::cout << "Press any key to close window . . ." << std::endl;
        //std::cin.get();
        return EXIT_FAILURE;
    }
}

void mainThread::correct_image(
    const ImageProcessing& ip,
    byte* slm_image_ptr, const byte* phasemap_ptr, const byte* phase_correction_ptr, const byte* lut_ptr
) {
    ip.expand_to_sensor_size(slm_image_ptr, phasemap_ptr);

    ip.add_blazed_grating(slm_image_ptr);

    ip.correct_image(slm_image_ptr, phase_correction_ptr, lut_ptr);

}
