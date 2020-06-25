#pragma once
// STL
#include "mainThread.h"
#include "ImageCapture.h"
#include <sstream>
#include "errorMessage.h"
#include "QApplication.h"


mainThread::mainThread(statusBox *box) {
    editM = box;
}

void mainThread::setDevice() {
    int count = 0;
    std::string GPU_name = "";
    std::string name0 = "";
    bool setDefaultDevice = true;
    cudaGetDeviceCount(&count);//expect two GPUs 
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        throw std::runtime_error("Could Not Find GPU Properties");
    }
    name0 = prop.name;
    for (int i = 0; i < count; i++) {
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
            throw std::runtime_error("Could Not Find GPU Properties");
        }
        GPU_name = (prop.name);
        if (GPU_name == "GeForce RTX 2080 SUPER") {
            setDefaultDevice = false;
            cudaSetDevice(i);//find the GPU we bought
            editM->appendMessage("Connected to GeForce RTX 2080 SUPER");
            printf("Connected to GeForce RTX 2080 SUPER");
            break;
        }
    }
    if (setDefaultDevice) {
        cudaSetDevice(0);//find the GPU we bought
        editM->appendMessage(("Connected to " + name0).c_str());
    }
}

void append(statusBox* box, const char* msg, std::string color = "black") {
    box->appendColorMessage(msg, color);
}//this is just for testing but I left some calls in main thread

int mainThread::run_thread() {
    try {
        editM->clear();
        setDevice();
        const auto params = Parameters();
        // define parameters to check their availability early on
        // This could be prevented by checking in params and using
        // default values if nothing is given

        const auto slm_px_x = params.get_slm_px_x();
        const auto slm_px_y = params.get_slm_px_y();
        const auto number_of_pixels_unpadded = params.get_number_of_pixels_unpadded();

        const auto camera_px_x = params.get_camera_px_x();
        const auto camera_px_y = params.get_camera_px_y();

        const auto lut_patch_size_x = params.get_lut_patch_size_x_px();
        const auto lut_patch_size_y = params.get_lut_patch_size_y_px();
        const auto lut_patch_num_x = params.get_number_of_lut_patches_x();
        const auto lut_patch_num_y = params.get_number_of_lut_patches_x();

        const auto num_traps_x = params.get_num_traps_x();
        const auto num_traps_y = params.get_num_traps_y();

        const double spacing_x_um = params.get_spacing_x_um();
        const double spacing_y_um = params.get_spacing_y_um();

        const double radial_shift_x_um = params.get_radial_shift_x_um();
        const double radial_shift_y_um = params.get_radial_shift_y_um();

        const double axial_shift_um = params.get_axial_shift_um();

        const double beam_waist_x_mm = params.get_beam_waist_x_mm();
        const double beam_waist_y_mm = params.get_beam_waist_y_mm();

        const std::string output_folder = params.get_output_folder();

        const auto frame_rate = params.get_frame_rate();

        // Camera feedback related variables
        // The ImageCapture instance is only created if feedback is enabled.
        // The program catches all ImageCaptureExceptions and terminates
        // in a controlled way

        std::unique_ptr<ImageCapture> ic_ptr;
        const bool camera_feedback_enabled = params.get_camera_feedback_enabled();

        if (camera_feedback_enabled) {
            editM->appendMessage("Camera feedback enabled");
            printf("Camera feedback enabled\n");
            try {
                ic_ptr = std::make_unique<ImageCapture>(params);

                // Check if camera feedback related variables are defined at this point so it
                // can terminate early on. It might be useful to add some functionality to Parameters
                // that checks if all array generation related variables have been defined.
                const size_t max_iterations_camera_feedback = params.get_max_iterations_camera_feedback();;
                const double max_nonuniformity_camera_feedback_percent = params.get_max_nonuniformity_camera_feedback_percent();
            }
            catch (const ImageCaptureException& e) {
                std::cout << e.what() << "\n";
                editM->appendColorMessage(e.what(), "red");
                // It seems that manually calling the params dtor
                // gives a faster cleanup
                params.~Parameters();
                //std::cout << "Press any key to close window . . ." << std::endl;
                //std::cin.get();
                return EXIT_FAILURE;
            }
        }


        const bool save_data = params.get_save_data();

        // Init before cuda stuff is allocated
        const auto ip = ImageProcessing(params);
        auto cgh = CGHAlgorithm(params, editM);
        auto tweezer_array = TweezerArray(params, editM);

        byte* slm_image_ptr;
        byte* phase_correction_ptr;
        byte* lut_ptr;
        byte* phasemap_ptr;

        {
            // Todo: Add checks
            if (cudaSuccess != cudaMallocManaged(&slm_image_ptr, (size_t)slm_px_x * slm_px_y * sizeof(byte))) {
                throw std::runtime_error("Couldn't Allocate slm_image_ptr"); errBox("Couldn't Allocate slm_image_ptr", __FILE__, __LINE__);
            }
            if (cudaSuccess != cudaMallocManaged(&phase_correction_ptr, (size_t)slm_px_x * slm_px_y * sizeof(byte))) {
                throw std::runtime_error("Couldn't Allocate phase_correction_ptr"); 
                errBox("Couldn't Allocate phase_correction_ptr", __FILE__, __LINE__);
            }
            if (cudaSuccess != cudaMallocManaged(&lut_ptr, (size_t)256 * lut_patch_num_x * lut_patch_num_y * sizeof(byte))) {
                throw std::runtime_error("Couldn't Allocate lut_ptr");
                errBox("Couldn't Allocate lut_ptr", __FILE__, __LINE__);
            }
            if (cudaSuccess != cudaMallocManaged(&phasemap_ptr, (size_t)slm_px_x * slm_px_y * sizeof(byte))) {
                throw std::runtime_error("Couldn't Allocate phasemap_ptr");
                errBox("Couldn't Allocate phasemap_ptr", __FILE__, __LINE__);
            }
            cuda_utils::cuda_synchronize(__FILE__,__LINE__);
        }

        basic_fileIO::load_LUT(lut_ptr, lut_patch_num_x, lut_patch_num_y);
        basic_fileIO::load_phase_correction(phase_correction_ptr, slm_px_x, slm_px_y);

        cuda_utils::cuda_synchronize(__FILE__, __LINE__);

        std::cout << "Read correction files\n";
        editM->appendMessage("Read correction files");

        const auto start = std::chrono::system_clock::now();

        // Theoretical iteration
        const auto non_uniformity_vec = cgh.AWGS2D_loop(tweezer_array, phasemap_ptr);
        ip.shift_fourier_image(phasemap_ptr, radial_shift_x_um, radial_shift_y_um);
        ip.fresnel_lens(phasemap_ptr, slm_px_y, slm_px_y, axial_shift_um);
        correct_image(ip, slm_image_ptr, phasemap_ptr, phase_correction_ptr, lut_ptr);

        // Save output
        if (save_data) {

            cgh.save_output_intensity_distribution_max(//max means Max's modification
                output_folder + "theory_output.bmp", tweezer_array
            );

            cgh.save_input_phase_distribution(
                output_folder + "phase_map.bmp"//saves square dimensions
            );

            basic_fileIO::save_one_column_data(
                output_folder + "nonuniformity.txt",
                non_uniformity_vec.cbegin(), non_uniformity_vec.cend()
            );

        }


        // Init window. The display is mostly stable
        // but it takes some time to open the window so it should be called
        // a few times
        /*{
            init_window(slm_px_x, slm_px_y, frame_rate);
            Sleep(500);
            size_t cnt = 0;
            while (cnt < 3) {
                display_phasemap(slm_image_ptr);
                Sleep(100);
                cnt++;
            }
        }*/

        if (camera_feedback_enabled) {
            camera_feedback_loop(
                params, ic_ptr, tweezer_array, cgh, ip,
                slm_image_ptr, phasemap_ptr, phase_correction_ptr, lut_ptr,
                number_of_pixels_unpadded
            );
        }



        const auto end = std::chrono::system_clock::now();
        const auto diff_in_s = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::stringstream stream;
        stream << "Iterating took: " << (diff_in_s / 60) << "min " << (diff_in_s % 60) << "s\n";
        editM->appendColorMessage(stream.str().c_str(),"green");
        std::cout << stream.str();
        stream.str(std::string());

        cudaFree(phasemap_ptr);
        cudaFree(phase_correction_ptr);
        cudaFree(lut_ptr);

        //display_phasemap(slm_image_ptr);
        basic_fileIO::save_as_bmp(output_folder + "phase_map.bmp", slm_image_ptr, params.get_slm_px_x(), params.get_slm_px_y());

        cudaFree(slm_image_ptr);

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


void mainThread::camera_feedback_loop(
    const Parameters& params, std::unique_ptr<ImageCapture>& ic_ptr,
    TweezerArray& tweezer_array, CGHAlgorithm& cgh, const ImageProcessing& ip,
    byte* slm_image_ptr,
    byte* phasemap_ptr,
    const byte* phase_correction_ptr,
    const byte* lut_ptr,
    unsigned int number_of_pixels_unpadded
) {
    const auto num_traps_x = params.get_num_traps_x();
    const auto num_traps_y = params.get_num_traps_y();

    const auto camera_px_x = params.get_camera_px_x();
    const auto camera_px_y = params.get_camera_px_y();
    const auto save_data = params.get_save_data();
    const auto output_folder = params.get_output_folder();

    const auto max_iterations_camera_feedback = params.get_max_iterations_camera_feedback();
    const auto max_nonuniformity_camera_feedback_percent = params.get_max_nonuniformity_camera_feedback_percent();

    const auto radial_shift_x_um = params.get_radial_shift_x_um();
    const auto radial_shift_y_um = params.get_radial_shift_y_um();

    const auto axial_shift_um = params.get_axial_shift_um();

    editM->appendMessage("\nStarting camera feedback\n");
    printf("\nStarting camera feedback\n\n");

    if (ic_ptr) {
        ic_ptr->adjust_exposure_time_automatically(230, 10, editM);
    }
    else {
        throw std::runtime_error("ic_ptr is null");
        errBox("ic_ptr ia null",__FILE__,__LINE__);
    }

    std::vector<byte> image_data(camera_px_x * camera_px_y);

    // Save before feedback
    if (save_data) {
        if (ic_ptr) {
            ic_ptr->capture_image(image_data.data(), camera_px_x, camera_px_y);
        }
        else {
            throw std::runtime_error("ic_ptr is null");
            errBox("ic_ptr ia null", __FILE__, __LINE__);
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
        }
        else {
            errBox("ic_ptr ia null", __FILE__, __LINE__);
            throw std::runtime_error("ic_ptr is null");
        }
        // Undo the shift
        ip.shift_fourier_image(phasemap_ptr, -radial_shift_x_um, -radial_shift_y_um);
        ip.fresnel_lens(phasemap_ptr, number_of_pixels_unpadded, number_of_pixels_unpadded, -axial_shift_um);

        // The camera image is flipped about both axis so we have to undo that
        ip.invert_camera_image(image_data.data(), camera_px_x, camera_px_y);

        // Because the optical system is jittering quite a lot peak positions can drift by up to a few px
        // between images. If the system is stabilized later this only needs to be done once in the beginning.
        const auto sorted_flattened_peak_indices = ip.create_mask(image_data.data(), camera_px_x, camera_px_y, num_traps_x, num_traps_y);

        tweezer_array.update_position_in_camera_image(sorted_flattened_peak_indices);

        delta = cgh.AWGS2D_camera_feedback_iteration(
            tweezer_array,
            image_data.data(),
            phasemap_ptr
        );

        non_uniformity_vec.push_back(delta);

        std::stringstream stream;
        stream << std::setfill('0') << std::setw(long long(log10(max_iterations_camera_feedback) + 1));
        stream << iteration + 1 << "/" << max_iterations_camera_feedback << "; ";
        stream << "Non-uniformity: " << std::setw(3) << 100 * delta << "%\n";

        editM->appendMessage(stream.str().c_str());
        std::cout << stream.str();
        stream.str(std::string());

        ip.shift_fourier_image(phasemap_ptr, radial_shift_x_um, radial_shift_y_um);
        ip.fresnel_lens(phasemap_ptr, number_of_pixels_unpadded, number_of_pixels_unpadded, axial_shift_um);

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
            errBox("ic_ptr ia null", __FILE__, __LINE__);
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
    }

    // Delete manually so that Vimba can be opened while the image is shown
    if (ic_ptr) {
        ic_ptr->~ImageCapture();
    }
    else {
        errBox("ic_ptr ia null", __FILE__, __LINE__);
        throw std::runtime_error("ic_ptr is null");
    }
}
