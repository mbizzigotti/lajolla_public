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
	if (device) vkDeviceWaitIdle(device);
	if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, 0);
	instance_buffer.Destroy(device);
	vertex_buffer.Destroy(device);
	index_buffer.Destroy(device);
	sbt_buffer.Destroy(device);
	tas.Destroy(*this);
	bas.Destroy(*this);
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
		LOAD_VULKAN_FUNCTION(vkGetBufferDeviceAddressKHR);
		LOAD_VULKAN_FUNCTION(vkCreateAccelerationStructureKHR);
		LOAD_VULKAN_FUNCTION(vkDestroyAccelerationStructureKHR);
		LOAD_VULKAN_FUNCTION(vkGetAccelerationStructureDeviceAddressKHR);
		LOAD_VULKAN_FUNCTION(vkCmdBuildAccelerationStructuresKHR);
		LOAD_VULKAN_FUNCTION(vkBuildAccelerationStructuresKHR);
		LOAD_VULKAN_FUNCTION(vkCreateRayTracingPipelinesKHR);
		LOAD_VULKAN_FUNCTION(vkCmdTraceRaysKHR);
		LOAD_VULKAN_FUNCTION(vkGetRayTracingShaderGroupHandlesKHR);
		LOAD_VULKAN_FUNCTION(vkGetAccelerationStructureBuildSizesKHR);
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
		for (VkSurfaceFormatKHR format : formats) {
			if (format.format == VK_FORMAT_R8G8B8A8_UNORM) {
				surfaceFormat = format;
			}
		}
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

	// Helper to create buffer + memory
	auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
		VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &bi, nullptr, &buf);
		VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, buf, &mr);
		VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize = mr.size;
		VkMemoryAllocateFlagsInfo af{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &af;
		ai.memoryTypeIndex = find_memory_type(physical_device, mr.memoryTypeBits, props);
		vkAllocateMemory(device, &ai, nullptr, &mem);
		vkBindBufferMemory(device, buf, mem, 0);
	};

	LOG("Creating Acceleration Structures...");
	{
		// Create vertex and index buffers for a single triangle
		struct Vertex { float x, y, z; };
		Vertex verts[3] = { {-0.5f,-0.5f,2.0f}, {0.5f,-0.5f,2.0f}, {0.0f,0.5f,2.0f} };
		uint32_t inds[3] = { 0,1,2 };

		createBuffer(sizeof(verts), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer.buffer, vertex_buffer.memory);
		createBuffer(sizeof(inds), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, index_buffer.buffer, index_buffer.memory);

		// staging buffers to upload data
		VkBuffer vStaging; VkDeviceMemory vStagingMem; createBuffer(sizeof(verts), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStaging, vStagingMem);
		VkBuffer iStaging; VkDeviceMemory iStagingMem; createBuffer(sizeof(inds), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStaging, iStagingMem);
		void* p; vkMapMemory(device, vStagingMem, 0, sizeof(verts), 0, &p); memcpy(p, verts, sizeof(verts)); vkUnmapMemory(device, vStagingMem);
		vkMapMemory(device, iStagingMem, 0, sizeof(inds), 0, &p); memcpy(p, inds, sizeof(inds)); vkUnmapMemory(device, iStagingMem);

		// copy staging -> device buffers
		VkCommandBufferAllocateInfo cba{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; cba.commandPool = command_pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
		VkCommandBuffer copyCmd; vkAllocateCommandBuffers(device, &cba, &copyCmd);
		VkCommandBufferBeginInfo binfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(copyCmd, &binfo);
		VkBufferCopy copyRegion{ 0,0,sizeof(verts) }; vkCmdCopyBuffer(copyCmd, vStaging, vertex_buffer, 1, &copyRegion);
		VkBufferCopy copyRegion2{ 0,0,sizeof(inds) }; vkCmdCopyBuffer(copyCmd, iStaging, index_buffer, 1, &copyRegion2);
		vkEndCommandBuffer(copyCmd);
		VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &copyCmd;
		vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);
		vkFreeCommandBuffers(device, command_pool, 1, &copyCmd);

		vkDestroyBuffer(device, vStaging, 0);
		vkFreeMemory(device, vStagingMem, 0);
		vkDestroyBuffer(device, iStaging, 0);
		vkFreeMemory(device, iStagingMem, 0);

		// Get device addresses
		VkBufferDeviceAddressInfo bufAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; bufAddrInfo.buffer = vertex_buffer;
		VkDeviceAddress vertexAddr = vkGetBufferDeviceAddress(device, &bufAddrInfo);
		bufAddrInfo.buffer = index_buffer; VkDeviceAddress indexAddr = vkGetBufferDeviceAddress(device, &bufAddrInfo);

		// Build BLAS (triangle)
		VkAccelerationStructureGeometryTrianglesDataKHR triangles{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR };
		triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
		triangles.vertexData.deviceAddress = vertexAddr;
		triangles.vertexStride = sizeof(Vertex);
		triangles.maxVertex = 3;
		triangles.indexType = VK_INDEX_TYPE_UINT32;
		triangles.indexData.deviceAddress = indexAddr;

		VkAccelerationStructureGeometryKHR geom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
		geom.geometry.triangles = triangles;

		VkAccelerationStructureBuildRangeInfoKHR rangeInfo{}; rangeInfo.primitiveCount = 1;
		const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

		VkAccelerationStructureBuildGeometryInfoKHR buildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		buildInfo.geometryCount = 1;
		buildInfo.pGeometries = &geom;

		uint32_t maxPrimitiveCounts[1] = { 1 };
		VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, maxPrimitiveCounts, &sizeInfo);

		// Create BLAS buffer
		createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bas.buffer.buffer, bas.buffer.memory);
		VkAccelerationStructureCreateInfoKHR asCreate{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
		asCreate.buffer = bas.buffer.buffer; asCreate.size = sizeInfo.accelerationStructureSize; asCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &asCreate, nullptr, &bas.handle);

		// scratch buffer
		VkBuffer scratch; VkDeviceMemory scratchMem; createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, scratchMem);
		VkBufferDeviceAddressInfo saddr{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; saddr.buffer = scratch; VkDeviceAddress scratchAddr = vkGetBufferDeviceAddress(device, &saddr);

		// Build BLAS command
		VkCommandBufferAllocateInfo ab{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; ab.commandPool = command_pool; ab.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ab.commandBufferCount = 1;
		VkCommandBuffer buildCmd; vkAllocateCommandBuffers(device, &ab, &buildCmd);
		vkBeginCommandBuffer(buildCmd, &binfo);
		buildInfo.dstAccelerationStructure = bas.handle;
		buildInfo.scratchData.deviceAddress = scratchAddr;
		vkCmdBuildAccelerationStructuresKHR(buildCmd, 1, &buildInfo, &pRange);
		vkEndCommandBuffer(buildCmd);
		si.commandBufferCount = 1; si.pCommandBuffers = &buildCmd;
		vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);
		vkFreeCommandBuffers(device, command_pool, 1, &buildCmd);

		vkDestroyBuffer(device, scratch, 0);
		vkFreeMemory(device, scratchMem, 0);

		// Get BLAS device address
		VkAccelerationStructureDeviceAddressInfoKHR addrInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
		addrInfo.accelerationStructure = bas.handle;
		bas.address = vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

		// Create TLAS (one instance referencing BLAS)
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = { { {1,0,0,0}, {0,1,0,0}, {0,0,1,0} } };
		asInstance.instanceCustomIndex = 0;
		asInstance.mask = 0xFF;
		asInstance.instanceShaderBindingTableRecordOffset = 0;
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		// set accelerationStructureReference
		uint64_t blasRef = bas.address;
		memcpy(&asInstance.accelerationStructureReference, &blasRef, sizeof(blasRef));

		// create instance buffer
		createBuffer(sizeof(asInstance), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instance_buffer.buffer, instance_buffer.memory);
		VkBuffer instStaging; VkDeviceMemory instStagingMem; createBuffer(sizeof(asInstance), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, instStaging, instStagingMem);
		vkMapMemory(device, instStagingMem, 0, sizeof(asInstance), 0, &p); memcpy(p, &asInstance, sizeof(asInstance)); vkUnmapMemory(device, instStagingMem);
		// copy
		vkAllocateCommandBuffers(device, &cba, &copyCmd);
		vkBeginCommandBuffer(copyCmd, &binfo);
		VkBufferCopy creg{ 0,0,sizeof(asInstance) }; vkCmdCopyBuffer(copyCmd, instStaging, instance_buffer, 1, &creg);
		vkEndCommandBuffer(copyCmd);
		si.pCommandBuffers = &copyCmd; vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(graphics_queue); vkFreeCommandBuffers(device, command_pool, 1, &copyCmd);

		vkDestroyBuffer(device, instStaging, 0);
		vkFreeMemory(device, instStagingMem, 0);

		// Build TLAS
		VkAccelerationStructureGeometryInstancesDataKHR instancesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		VkBufferDeviceAddressInfo instAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; instAddrInfo.buffer = instance_buffer; instancesData.data.deviceAddress = vkGetBufferDeviceAddress(device, &instAddrInfo);
		VkAccelerationStructureGeometryKHR iGeom{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
		iGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR; iGeom.geometry.instances = instancesData;

		VkAccelerationStructureBuildGeometryInfoKHR tBuildInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
		tBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR; tBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		tBuildInfo.geometryCount = 1; tBuildInfo.pGeometries = &iGeom;
		uint32_t maxPrimCountsTLAS[1] = { 1 };
		VkAccelerationStructureBuildSizesInfoKHR tSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tBuildInfo, maxPrimCountsTLAS, &tSizes);

		createBuffer(tSizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tas.buffer.buffer, tas.buffer.memory);
		VkAccelerationStructureCreateInfoKHR tcreate{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR }; tcreate.buffer = tas.buffer.buffer; tcreate.size = tSizes.accelerationStructureSize; tcreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &tcreate, nullptr, &tas.handle);

		// scratch for TLAS
		VkBuffer tscratch; VkDeviceMemory tscratchMem; createBuffer(tSizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tscratch, tscratchMem);
		saddr.buffer = tscratch; scratchAddr = vkGetBufferDeviceAddress(device, &saddr);

		// build TLAS command
		assert(vkAllocateCommandBuffers(device, &ab, &buildCmd) == VK_SUCCESS);
		assert(vkBeginCommandBuffer(buildCmd, &binfo) == VK_SUCCESS);
		tBuildInfo.dstAccelerationStructure = tas.handle;
		tBuildInfo.scratchData.deviceAddress = scratchAddr;
		vkCmdBuildAccelerationStructuresKHR(buildCmd, 1, &tBuildInfo, &pRange);
		assert(vkEndCommandBuffer(buildCmd) == VK_SUCCESS);
		si.pCommandBuffers = &buildCmd;
		assert(vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
		assert(vkQueueWaitIdle(graphics_queue) == VK_SUCCESS);
		vkFreeCommandBuffers(device, command_pool, 1, &buildCmd);

		vkDestroyBuffer(device, tscratch, 0);
		vkFreeMemory(device, tscratchMem, 0);
	}
	LOG("Creating Shader Binding Table...");
	{
		VkPhysicalDeviceProperties2 pdprops{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
		VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtprops{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
		pdprops.pNext = &rtprops;
		vkGetPhysicalDeviceProperties2(physical_device, &pdprops);
		uint32_t handleSize = rtprops.shaderGroupHandleSize;
		uint32_t baseAlignment = rtprops.shaderGroupBaseAlignment;

		uint32_t groupCount = 3;
		std::vector<char> shaderHandleStorage(groupCount * handleSize);
		vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount, shaderHandleStorage.size(), shaderHandleStorage.data());

		// create SBT buffer (host visible)
		VkDeviceSize sbtSize = groupCount * baseAlignment;
		createBuffer(sbtSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sbt_buffer.buffer, sbt_buffer.memory);
		void* sbtMap; vkMapMemory(device, sbt_buffer.memory, 0, sbtSize, 0, &sbtMap);
		for (uint32_t g = 0; g < groupCount; ++g) {
			memcpy(reinterpret_cast<char*>(sbtMap) + g * baseAlignment, shaderHandleStorage.data() + g * handleSize, handleSize);
		}
		vkUnmapMemory(device, sbt_buffer.memory);
		VkBufferDeviceAddressInfo sbtAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; sbtAddrInfo.buffer = sbt_buffer; VkDeviceAddress sbtAddr = vkGetBufferDeviceAddress(device, &sbtAddrInfo);

		rgen_sbt = { .deviceAddress = sbtAddr + 0 * baseAlignment, .stride = baseAlignment, .size = baseAlignment };
		miss_sbt = { .deviceAddress = sbtAddr + 1 * baseAlignment, .stride = baseAlignment, .size = baseAlignment };
		chit_sbt = { .deviceAddress = sbtAddr + 2 * baseAlignment, .stride = baseAlignment, .size = baseAlignment };
	}
	LOG("Creating Descriptor Set...");
	{
		// Descriptor pool and set
		VkDescriptorPoolSize poolSizes[2]; poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; poolSizes[0].descriptorCount = 1; poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; poolSizes[1].descriptorCount = 1;
		VkDescriptorPoolCreateInfo dpc{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO }; dpc.maxSets = 1; dpc.poolSizeCount = 2; dpc.pPoolSizes = poolSizes;
		assert(vkCreateDescriptorPool(device, &dpc, 0, &descriptor_pool) == VK_SUCCESS);
		VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; dsa.descriptorPool = descriptor_pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &descriptor_set_layout;
		assert(vkAllocateDescriptorSets(device, &dsa, &descriptor_set) == VK_SUCCESS);

		// Update descriptor with TLAS
		VkWriteDescriptorSetAccelerationStructureKHR asWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR }; asWrite.accelerationStructureCount = 1; VkAccelerationStructureKHR asList[] = { tas.handle }; asWrite.pAccelerationStructures = asList;
		VkWriteDescriptorSet wds{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; wds.dstSet = descriptor_set; wds.dstBinding = 0; wds.descriptorCount = 1; wds.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; wds.pNext = &asWrite;

		// Update storage image descriptor
		VkDescriptorImageInfo dii{}; dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL; dii.imageView = storage_view; dii.sampler = VK_NULL_HANDLE;
		VkWriteDescriptorSet wds2{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; wds2.dstSet = descriptor_set; wds2.dstBinding = 1; wds2.descriptorCount = 1; wds2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wds2.pImageInfo = &dii;

		VkWriteDescriptorSet writes[] = { wds, wds2 };
		vkUpdateDescriptorSets(device, (uint32_t)std::size(writes), writes, 0, 0);
	}
}

void GPUDevice::render(RGFW_window* window)
{
	// record command buffer: trace rays into storage image, then copy to swapchain image
	VkCommandBufferAllocateInfo ca{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; ca.commandPool = command_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
	VkCommandBuffer cmd; vkAllocateCommandBuffers(device, &ca, &cmd);

	uint32_t width = image_extent.width;
	uint32_t height = image_extent.height;

	while (!RGFW_window_shouldClose(window))
	{
		RGFW_pollEvents();

		uint32_t image_index;
		vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX, image_available_semaphore, 0, &image_index);

		VkCommandBufferBeginInfo bi2{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi2);

		// transition storage image to general
		VkImageMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.image = storage_image; barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; barrier.subresourceRange.baseMipLevel = 0; barrier.subresourceRange.levelCount = 1; barrier.subresourceRange.baseArrayLayer = 0; barrier.subresourceRange.layerCount = 1; barrier.srcAccessMask = 0; barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		// bind pipeline and descriptor sets and trace
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

		VkStridedDeviceAddressRegionKHR callable_sbt{};
		vkCmdTraceRaysKHR(cmd, &rgen_sbt, &miss_sbt, &chit_sbt, &callable_sbt, width, height, 1);

		// transition swapchain image to transfer dst
		VkImage dst = images[image_index];
		VkImageMemoryBarrier toCopy{}; toCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; toCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; toCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toCopy.image = dst; toCopy.subresourceRange = barrier.subresourceRange; toCopy.srcAccessMask = 0; toCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toCopy);

		// copy storage image -> swapchain image
		VkImageCopy imcopy{}; imcopy.srcOffset = { 0,0,0 }; imcopy.dstOffset = { 0,0,0 }; imcopy.extent = { width, height, 1 }; imcopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; imcopy.srcSubresource.layerCount = 1; imcopy.dstSubresource = imcopy.srcSubresource;
		vkCmdCopyImage(cmd, storage_image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imcopy);

		// transition swapchain image to present
		VkImageMemoryBarrier toPresent = toCopy; toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; toPresent.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; toPresent.dstAccessMask = 0;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &toPresent);

		vkEndCommandBuffer(cmd);

		VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; VkSemaphore waitSem[] = { image_available_semaphore }; VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
		submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = waitSem; submit.pWaitDstStageMask = waitStages; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd; VkSemaphore sig[] = { render_finished_semaphore }; submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = sig;
		vkQueueSubmit(graphics_queue, 1, &submit, VK_NULL_HANDLE);

		VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR }; pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = sig; pi.swapchainCount = 1; pi.pSwapchains = &swap_chain; pi.pImageIndices = &image_index;
		vkQueuePresentKHR(present_queue, &pi);
		vkQueueWaitIdle(present_queue);
	}
}

void VulkanAccelerationStructure::Destroy(struct GPUDevice& gpu)
{
	if (handle) gpu.vkDestroyAccelerationStructureKHR(gpu.device, handle, 0);
	buffer.Destroy(gpu.device);
}

void VulkanBuffer::Destroy(VkDevice device)
{
	if (buffer) vkDestroyBuffer(device, buffer, 0);
	if (memory) vkFreeMemory(device, memory, 0);
}
