#include "parsers/parse_scene.h"
#include "parallel.h"
#include "image.h"
#include "render.h"
#include "timer.h"
#include "gpu_device.h"
#include <embree4/rtcore.h>
#include "3rdparty/stb_image.h"
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "[Usage] ./lajolla [-g] [-t num_threads] [-o output_file_name] filename.xml" << std::endl;
        return 0;
	}

	bool use_gpu = false;
    int num_threads = std::thread::hardware_concurrency();
    std::string outputfile = "";
    std::vector<std::string> filenames;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-t") {
            num_threads = std::stoi(std::string(argv[++i]));
        } else if (std::string(argv[i]) == "-o") {
            outputfile = std::string(argv[++i]);
        } else if (std::string(argv[i]) == "-g") {
			use_gpu = true;
        } else {
            filenames.push_back(std::string(argv[i]));
        }
    }

	if (use_gpu) {
		GPUDevice gpu_device;
        Timer timer;
        tick(timer);
        std::cout << "Parsing and constructing scene " << filenames[0] << "." << std::endl;
        std::unique_ptr<Scene> scene = parse_scene(filenames[0], &gpu_device);
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        std::cout << "Rendering..." << std::endl;
		RGFW_window *window = RGFW_createWindow(
			"LaJolla!", 0, 0, scene->camera.width, scene->camera.height,
            RGFW_windowCenter | RGFW_windowNoResize);
        { // Set Icon because why not?
            int w, h, comp;
            u8* pixels = stbi_load("icon.png", &w, &h, &comp, 4);
            if (pixels) RGFW_window_setIcon(window, pixels, w, h, RGFW_formatRGBA8);
        }
		gpu_device.attach(window, scene.get());
		gpu_device.render(window);
		return 0;
	}

    RTCDevice embree_device = rtcNewDevice(nullptr);
    parallel_init(num_threads);

    for (const std::string &filename : filenames) {
        Timer timer;
        tick(timer);
        std::cout << "Parsing and constructing scene " << filename << "." << std::endl;
        std::unique_ptr<Scene> scene = parse_scene(filename, embree_device);
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        std::cout << "Rendering..." << std::endl;
        Image3 img = render(*scene);
        if (outputfile.compare("") == 0) {outputfile = scene->output_filename;}
        std::cout << "Done. Took " << tick(timer) << " seconds." << std::endl;
        imwrite(outputfile, img);
        std::cout << "Image written to " << outputfile << std::endl;
    }

    parallel_cleanup();
    rtcReleaseDevice(embree_device);
    return 0;
}
