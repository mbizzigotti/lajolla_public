#include "gpu_device.h"

#define LOG(...) printf(__VA_ARGS__); putchar('\n')

GPUDevice::GPUDevice()
{
	LOG("Creating Vulkan Instance...");
	{
		VkApplicationInfo application_info {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = "LaJolla!",
			.applicationVersion = VK_MAKE_API_VERSION(1, 0, 0, 1),
			.pEngineName = "🏄‍♂️",
			.engineVersion = VK_MAKE_API_VERSION(1, 0, 0, 1),
			.apiVersion = VK_API_VERSION_1_2,
		};

		const char* extensions[] = {
			VK_KHR_SURFACE_EXTENSION_NAME,
			RGFW_VK_SURFACE,
		};

		VkInstanceCreateInfo createInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &application_info,
			.enabledExtensionCount = (uint32_t)std::size(extensions),
			.ppEnabledExtensionNames = extensions,
		};
		assert(vkCreateInstance(&createInfo, 0, &instance) == VK_SUCCESS);
	}
}

GPUDevice::~GPUDevice()
{
	if (instance) vkDestroyInstance(instance, 0);
}

void GPUDevice::attach(RGFW_window* window, Scene* scene)
{
	LOG("TODO");
}

void GPUDevice::render(RGFW_window* window)
{
	LOG("TODO");
}
