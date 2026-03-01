#include "gpu_device.h"

#ifdef ERROR
#undef ERROR
#endif
#define LOG(...) printf(__VA_ARGS__), putchar('\n')
#define ERROR(...) LOG(__VA_ARGS__), exit(1)
#define LOAD_VULKAN_FUNCTION(NAME) \
	assert(NAME = (PFN_##NAME)vkGetDeviceProcAddr(device, #NAME));

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	bool isComplete() const { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
	QueueFamilyIndices indices;
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
	std::vector<VkQueueFamilyProperties> families(count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

	for (uint32_t i = 0; i < count; ++i) {
		if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) indices.graphicsFamily = i;
		VkBool32 presentSupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (presentSupport) indices.presentFamily = i;
		if (indices.isComplete()) break;
	}
	return indices;
}

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

		const char* layers[] = {
			"VK_LAYER_KHRONOS_validation"
		};

		VkInstanceCreateInfo createInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &application_info,
			.enabledLayerCount = (uint32_t)std::size(layers),
			.ppEnabledLayerNames = layers,
			.enabledExtensionCount = (uint32_t)std::size(extensions),
			.ppEnabledExtensionNames = extensions,
		};
		assert(vkCreateInstance(&createInfo, 0, &instance) == VK_SUCCESS);
	}
}

GPUDevice::~GPUDevice()
{
	for (int i = 0; i < image_count; ++i) {
		vkDestroyImageView(device, image_views[i], 0);
	}
	if (swap_chain) vkDestroySwapchainKHR(device, swap_chain, 0);
	if (device) vkDestroyDevice(device, 0);
	if (surface) vkDestroySurfaceKHR(instance, surface, 0);
	if (instance) vkDestroyInstance(instance, 0);
}

void GPUDevice::attach(RGFW_window* window, Scene* scene)
{
	QueueFamilyIndices indices;

	LOG("Creating Window Surface...");
	{
		assert(RGFW_window_createSurface_Vulkan(window, instance, &surface) == VK_SUCCESS);
	}
	LOG("Picking a Physical Device that supports ray tracing...");
	{
		uint32_t device_count = 0;
		assert(vkEnumeratePhysicalDevices(instance, &device_count, 0) == VK_SUCCESS);
		assert(device_count > 0);
		std::vector<VkPhysicalDevice> devices(device_count);
		assert(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) == VK_SUCCESS);

		for (auto dev : devices) {
			indices = findQueueFamilies(dev, surface);
			if (!indices.isComplete()) continue;
			// check for ray tracing device extensions
			uint32_t extCount = 0;
			vkEnumerateDeviceExtensionProperties(dev, 0, &extCount, 0);
			std::vector<VkExtensionProperties> exts(extCount);
			vkEnumerateDeviceExtensionProperties(dev, 0, &extCount, exts.data());
			bool ok = true;
			const std::vector<const char*> required = {
				VK_KHR_SWAPCHAIN_EXTENSION_NAME,
				VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
				VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
				VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
				VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
				VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME
			};
			for (auto r : required) {
				bool found = false;
				for (auto& e : exts) if (strcmp(e.extensionName, r) == 0) { found = true; break; }
				if (!found) { ok = false; break; }
			}
			if (ok) { physical_device = dev; break; }
		}
		if (physical_device == VK_NULL_HANDLE) {
			ERROR("No physical device with ray tracing support found.");
		}
	}
	LOG("Creating Logical Device...");
	{
		VkPhysicalDeviceFeatures2                        device_features                 { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
		VkPhysicalDeviceBufferDeviceAddressFeatures      device_address_features         { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES };
		VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR    ray_tracing_features            { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };

		device_features                 .pNext = &device_address_features;
		device_address_features         .pNext = &acceleration_structure_features;
		acceleration_structure_features .pNext = &ray_tracing_features;

		auto vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2");
		assert(vkGetPhysicalDeviceFeatures2);
		vkGetPhysicalDeviceFeatures2(physical_device, &device_features);

		// request enabling of required features (if supported)
		device_address_features.bufferDeviceAddress = VK_TRUE;
		acceleration_structure_features.accelerationStructure = VK_TRUE;
		ray_tracing_features.rayTracingPipeline = VK_TRUE;

		// Create logical device with queues and feature pNext
		float priority = 1.0f;
		uint32_t queue_count = 0;
		VkDeviceQueueCreateInfo queue_infos[2];

		auto add_queue_family = [&](uint32_t index) {
			queue_infos[queue_count++] = VkDeviceQueueCreateInfo{
				.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
				.queueFamilyIndex = index,
				.queueCount = 1,
				.pQueuePriorities = &priority,
			};
		};

		add_queue_family(indices.graphicsFamily.value());
		if (indices.presentFamily.value() != indices.graphicsFamily.value())
			add_queue_family(indices.presentFamily.value());

		const char* extensions[] = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
			VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
			VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
			VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		};

		VkDeviceCreateInfo device_info {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &device_features,
			.queueCreateInfoCount = queue_count,
			.pQueueCreateInfos = queue_infos,
			.enabledExtensionCount = (uint32_t)std::size(extensions),
			.ppEnabledExtensionNames = extensions,
		};
		assert(vkCreateDevice(physical_device, &device_info, 0, &device) == VK_SUCCESS);

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphics_queue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &present_queue);
	}
	LOG("Loading Ray Tracing API Functions...");
	{
		// Load ray tracing function pointers
		LOAD_VULKAN_FUNCTION(vkGetBufferDeviceAddressKHR);
		LOAD_VULKAN_FUNCTION(vkCreateAccelerationStructureKHR);
		LOAD_VULKAN_FUNCTION(vkDestroyAccelerationStructureKHR);
		LOAD_VULKAN_FUNCTION(vkGetAccelerationStructureDeviceAddressKHR);
		LOAD_VULKAN_FUNCTION(vkCmdBuildAccelerationStructuresKHR);
		LOAD_VULKAN_FUNCTION(vkBuildAccelerationStructuresKHR);
		LOAD_VULKAN_FUNCTION(vkCreateRayTracingPipelinesKHR);
		LOAD_VULKAN_FUNCTION(vkCmdTraceRaysKHR);
		LOAD_VULKAN_FUNCTION(vkGetRayTracingShaderGroupHandlesKHR);
	}
	LOG("Creating Swap Chain...");
	{
		VkSurfaceCapabilitiesKHR capabilities = {};
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);
		uint32_t format_count = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nullptr);
		assert(format_count > 0);
		std::vector<VkSurfaceFormatKHR> formats(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, formats.data());
		VkSurfaceFormatKHR surfaceFormat = formats[0];
		swap_format = surfaceFormat.format;

		VkExtent2D extent = { (uint32_t)scene->camera.width, (uint32_t)scene->camera.height };
		uint32_t imageCount = capabilities.minImageCount + 1;
		if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
			imageCount = capabilities.maxImageCount;

		uint32_t families[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		VkSwapchainCreateInfoKHR swap_chain_info {
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = imageCount,
			.imageFormat = surfaceFormat.format,
			.imageColorSpace = surfaceFormat.colorSpace,
			.imageExtent = extent,
			.imageArrayLayers = 1,
			.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.preTransform = capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_FIFO_KHR,
			.clipped = VK_TRUE,
		};
		if (indices.graphicsFamily.value() != indices.presentFamily.value()) {
			swap_chain_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			swap_chain_info.queueFamilyIndexCount = 2;
			swap_chain_info.pQueueFamilyIndices = families;
		}
		assert(vkCreateSwapchainKHR(device, &swap_chain_info, 0, &swap_chain) == VK_SUCCESS);
	}
	LOG("Creating Swap Chain Image Views...");
	{
		assert(vkGetSwapchainImagesKHR(device, swap_chain, &image_count, 0) == VK_SUCCESS);
		assert(image_count > 0);
		assert(image_count <= MAX_SWAP_CHAIN_IMAGES);
		assert(vkGetSwapchainImagesKHR(device, swap_chain, &image_count, images) == VK_SUCCESS);

		VkImageViewCreateInfo view_info {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = swap_format,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		for (int i = 0; i < image_count; ++i) {
			view_info.image = images[i];
			assert(vkCreateImageView(device, &view_info, 0, &image_views[i]) == VK_SUCCESS);
		}
	}
	LOG("TODO");
}

void GPUDevice::render(RGFW_window* window)
{
	LOG("TODO");
}
