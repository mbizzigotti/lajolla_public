#include "gpu_device.h"
#include <fstream>

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

VkExtent3D extent3d(VkExtent2D extent2d) {
	return { extent2d.width, extent2d.height, 1 };
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
	for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
		if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	assert(false && "Failed to find memory type");
	return 0;
}

static VkShaderModule load_shader_from_file(VkDevice device, const char* path) {
	LOG(" ... \"%s\"", path);
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	assert(file && "Failed to open shader file");
	size_t size = (size_t)file.tellg();
	std::vector<char> buf(size);
	file.seekg(0);
	file.read(buf.data(), size);

	VkShaderModuleCreateInfo shader_info {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 
		.codeSize = buf.size(),
		.pCode = reinterpret_cast<const uint32_t*>(buf.data()),
	};
	VkShaderModule shader{ 0 };
	assert(vkCreateShaderModule(device, &shader_info, 0, &shader) == VK_SUCCESS);
	return shader;
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
			.apiVersion = VK_API_VERSION_1_3,
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
	if (pipeline) vkDestroyPipeline(device, pipeline, 0);
	if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, 0);
	if (descriptor_set_layout) vkDestroyDescriptorSetLayout(device, descriptor_set_layout, 0);
	if (storage_view) vkDestroyImageView(device, storage_view, 0);
	if (storage_image) vkDestroyImage(device, storage_image, 0);
	if (storage_memory) vkFreeMemory(device, storage_memory, 0);
	if (render_finished_semaphore) vkDestroySemaphore(device, render_finished_semaphore, 0);
	if (image_available_semaphore) vkDestroySemaphore(device, image_available_semaphore, 0);
	if (command_pool) vkDestroyCommandPool(device, command_pool, 0);
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

		image_extent = { (uint32_t)scene->camera.width, (uint32_t)scene->camera.height };
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
			.imageExtent = image_extent,
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
	LOG("Creating Command Pool and Sync. Primitives...");
	{
		VkCommandPoolCreateInfo pool_info {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = indices.graphicsFamily.value(),
		};
		assert(vkCreateCommandPool(device, &pool_info, 0, &command_pool) == VK_SUCCESS);

		VkSemaphoreCreateInfo semaphore_info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_semaphore);
		vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphore);
	}
	LOG("Creating Storage Image...");
	{
		VkImageCreateInfo imgInfo{
			.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
			.imageType = VK_IMAGE_TYPE_2D,
			.format = VK_FORMAT_R8G8B8A8_UNORM,
			.extent = extent3d(image_extent),
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		assert(vkCreateImage(device, &imgInfo, nullptr, &storage_image) == VK_SUCCESS);
		
		VkMemoryRequirements memReq;
		vkGetImageMemoryRequirements(device, storage_image, &memReq);
		
		VkMemoryAllocateInfo ainfo { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
		ainfo.allocationSize = memReq.size;
		ainfo.memoryTypeIndex = find_memory_type(physical_device, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		vkAllocateMemory(device, &ainfo, nullptr, &storage_memory);
		vkBindImageMemory(device, storage_image, storage_memory, 0);

		VkImageViewCreateInfo siv{}; siv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		siv.image = storage_image; siv.viewType = VK_IMAGE_VIEW_TYPE_2D; siv.format = VK_FORMAT_R8G8B8A8_UNORM;
		siv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; siv.subresourceRange.baseMipLevel = 0; siv.subresourceRange.levelCount = 1;
		siv.subresourceRange.baseArrayLayer = 0; siv. subresourceRange.layerCount = 1;
		vkCreateImageView(device, &siv, nullptr, &storage_view);
	}
	LOG("Creating Layouts...");
	{
		// Descriptor set: acceleration structure and storage image
		VkDescriptorSetLayoutBinding asBinding{}; asBinding.binding = 0; asBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; asBinding.descriptorCount = 1; asBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		VkDescriptorSetLayoutBinding imgBinding{}; imgBinding.binding = 1; imgBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; imgBinding.descriptorCount = 1; imgBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		std::vector<VkDescriptorSetLayoutBinding> bindings = { asBinding, imgBinding };
		VkDescriptorSetLayoutCreateInfo dsl{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO }; dsl.bindingCount = (uint32_t)bindings.size(); dsl.pBindings = bindings.data();
		assert(vkCreateDescriptorSetLayout(device, &dsl, 0, &descriptor_set_layout) == VK_SUCCESS);

		VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO }; plci.setLayoutCount = 1; plci.pSetLayouts = &descriptor_set_layout;
		assert(vkCreatePipelineLayout(device, &plci, 0, &pipeline_layout) == VK_SUCCESS);
	}

	VkShaderModule shader_rgen{ 0 };
	VkShaderModule shader_miss{ 0 };
	VkShaderModule shader_chit{ 0 };

	LOG("Loading Shaders...");
	{
		shader_rgen = load_shader_from_file(device, "shaders/raygen.spv");
		shader_miss = load_shader_from_file(device, "shaders/miss.spv");
		shader_chit = load_shader_from_file(device, "shaders/chit.spv");
	}
	LOG("Creating Ray Tracing Pipeline...");
	{
		VkPipelineShaderStageCreateInfo stages[] {
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
				.module = shader_rgen,
				.pName = "main",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_MISS_BIT_KHR,
				.module = shader_miss,
				.pName = "main",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
				.module = shader_chit,
				.pName = "main",
			},
		};

		// Shader groups: raygen(0), miss(1), hitgroup(2)
		VkRayTracingShaderGroupCreateInfoKHR groups[] {
			{
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
				.generalShader      = 0,
				.closestHitShader   = VK_SHADER_UNUSED_KHR,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			},
			{
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
				.generalShader      = 1,
				.closestHitShader   = VK_SHADER_UNUSED_KHR,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			},
			{
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
				.generalShader      = VK_SHADER_UNUSED_KHR,
				.closestHitShader   = 2,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			},
		};

		VkRayTracingPipelineCreateInfoKHR pipeline_info {
			.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
			.stageCount = (uint32_t)std::size(stages),
			.pStages = stages,
			.groupCount = (uint32_t)std::size(groups),
			.pGroups = groups,
			.maxPipelineRayRecursionDepth = 1,
			.layout = pipeline_layout,
		};
		assert(vkCreateRayTracingPipelinesKHR(device, 0, 0, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS);
	}

	vkDestroyShaderModule(device, shader_rgen, 0);
	vkDestroyShaderModule(device, shader_miss, 0);
	vkDestroyShaderModule(device, shader_chit, 0);

	LOG("TODO");
}

void GPUDevice::render(RGFW_window* window)
{
	LOG("TODO");
}
