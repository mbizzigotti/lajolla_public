#include "gpu_device.h"
#include "shaders/shared.slang"
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

static VkShaderModule load_shader_from_file(VkDevice device, const char* path) {
	LOG(" ... \"%s\"", path);
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	assert(file.is_open() && "Failed to open shader file");
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

VkDeviceAddress get_buffer_device_address(VkDevice device, VkBuffer buffer) {
	VkBufferDeviceAddressInfo info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer,
	};
	return vkGetBufferDeviceAddress(device, &info);
}

void GPUDevice::add_shape(const Shape &shape) {
	VulkanTriangleMesh triangle_mesh{};

	const TriangleMesh &mesh = std::get<TriangleMesh>(shape);
	TVector3<float>* positions = new TVector3<float>[mesh.positions.size()];

	for (int i = 0; i < mesh.positions.size(); ++i)
		positions[i] = TVector3<float>(mesh.positions[i]);

	size_t vertex_size = mesh.positions.size() * sizeof(TVector3<float>);
	size_t index_size = mesh.indices.size() * sizeof(Vector3i);

	createBuffer(vertex_size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, triangle_mesh.vertex_buffer.buffer, triangle_mesh.vertex_buffer.memory);
	createBuffer(index_size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, triangle_mesh.index_buffer.buffer, triangle_mesh.index_buffer.memory);

	// staging buffers to upload data
	VkBuffer vStaging; VkDeviceMemory vStagingMem; createBuffer(vertex_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStaging, vStagingMem);
	VkBuffer iStaging; VkDeviceMemory iStagingMem; createBuffer(index_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStaging, iStagingMem);
	void* p; vkMapMemory(device, vStagingMem, 0, vertex_size, 0, &p); memcpy(p, positions, vertex_size); vkUnmapMemory(device, vStagingMem);
	vkMapMemory(device, iStagingMem, 0, index_size, 0, &p); memcpy(p, mesh.indices.data(), index_size); vkUnmapMemory(device, iStagingMem);

	VkCommandBuffer cmd = temp_command_buffer();
	VkBufferCopy copyRegion{ 0,0,vertex_size }; vkCmdCopyBuffer(cmd, vStaging, triangle_mesh.vertex_buffer, 1, &copyRegion);
	VkBufferCopy copyRegion2{ 0,0,index_size }; vkCmdCopyBuffer(cmd, iStaging, triangle_mesh.index_buffer, 1, &copyRegion2);
	flush_and_destroy_command_buffer(cmd);

	vkDestroyBuffer(device, vStaging, 0);
	vkFreeMemory(device, vStagingMem, 0);
	vkDestroyBuffer(device, iStaging, 0);
	vkFreeMemory(device, iStagingMem, 0);

	VkAccelerationStructureGeometryKHR geometry = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {
			.triangles = {
				.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
				.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
				.vertexData = {.deviceAddress = get_buffer_device_address(device, triangle_mesh.vertex_buffer)},
				.vertexStride = sizeof(TVector3<float>),
				.maxVertex = (uint32_t)mesh.positions.size(),
				.indexType = VK_INDEX_TYPE_UINT32,
				.indexData = {.deviceAddress = get_buffer_device_address(device, triangle_mesh.index_buffer)},
			}
		},
		.flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
	};

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.geometryCount = 1,
		.pGeometries = &geometry,
	};
	uint32_t primitive_count = mesh.indices.size();
	VkAccelerationStructureBuildRangeInfoKHR build_range {
		.primitiveCount = primitive_count,
	};
	const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &build_range;

	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primitive_count, &sizeInfo);

	VulkanAccelerationStructure bas;

	// Create BLAS buffer
	createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, bas.buffer.buffer, bas.buffer.memory);
	VkAccelerationStructureCreateInfoKHR asCreate{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.buffer = bas.buffer.buffer,
		.size = sizeInfo.accelerationStructureSize,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
	};
	assert(vkCreateAccelerationStructureKHR(device, &asCreate, 0, &bas.handle) == VK_SUCCESS);

	// scratch buffer
	VkBuffer scratch; VkDeviceMemory scratchMem; createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, scratchMem);
	VkBufferDeviceAddressInfo saddr{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; saddr.buffer = scratch; VkDeviceAddress scratchAddr = vkGetBufferDeviceAddress(device, &saddr);

	// Build BLAS command
	{
		VkCommandBuffer cmd = temp_command_buffer();
		buildInfo.dstAccelerationStructure = bas.handle;
		buildInfo.scratchData.deviceAddress = scratchAddr;
		vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRanges);
		flush_and_destroy_command_buffer(cmd);
	}

	vkDestroyBuffer(device, scratch, 0);
	vkFreeMemory(device, scratchMem, 0);

	// Get BLAS device address
	VkAccelerationStructureDeviceAddressInfoKHR addrInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR };
	addrInfo.accelerationStructure = bas.handle;
	bas.address = vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

	// Create instance referencing BLAS
	VkAccelerationStructureInstanceKHR asInstance = {
		.transform = { { {1,0,0,0}, {0,1,0,0}, {0,0,1,0} } },
		.instanceCustomIndex = (uint32_t)mesh.material_id,
		.mask = 0xFF,
		.instanceShaderBindingTableRecordOffset = 0,
		.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
		.accelerationStructureReference = bas.address,
	};

	bass.emplace_back(bas);
	instances.emplace_back(asInstance);
	triangle_meshes.emplace_back(triangle_mesh);
}

struct filter_convert_op {
	GPU::Filter operator()(const Box& filter) const { return { GPU::Box, static_cast<float>(filter.width) }; }
	GPU::Filter operator()(const Tent& filter) const { return { GPU::Tent, static_cast<float>(filter.width) }; }
	GPU::Filter operator()(const Gaussian& filter) const { return { GPU::Gaussian, static_cast<float>(filter.stddev) }; }
};

GPU::Camera convert(const Camera &camera) {
	GPU::Camera out;
    out.sample_to_cam = TMatrix4x4<float>(camera.sample_to_cam);
	out.cam_to_sample = TMatrix4x4<float>(camera.cam_to_sample);
    out.cam_to_world = TMatrix4x4<float>(camera.cam_to_world);
	out.world_to_cam = TMatrix4x4<float>(camera.world_to_cam);
    out.width = camera.width;
	out.height = camera.height;
    out.filter = std::visit(filter_convert_op{}, camera.filter);
    out.medium_id = camera.medium_id;
	return out;
}

struct material_convert_op {
	void operator()(const Lambertian& bsdf) {
		GPU::Lambertian material;
		material.reflectance = std::get<ConstantTexture<Spectrum>>(bsdf.reflectance).value;
		raw.Add(material);
	}
	void operator()(const RoughPlastic& bsdf) { assert(false); }
	void operator()(const RoughDielectric& bsdf) { assert(false); }
	void operator()(const DisneyDiffuse& bsdf) { assert(false); }
	void operator()(const DisneyMetal& bsdf) { assert(false); }
	void operator()(const DisneyGlass& bsdf) { assert(false); }
	void operator()(const DisneyClearcoat& bsdf) { assert(false); }
	void operator()(const DisneySheen& bsdf) { assert(false); }
	void operator()(const DisneyBSDF& bsdf) { assert(false); }

	VulkanRawBuffer& raw;
};

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

		VkValidationFeatureEnableEXT validation_feature_enables[] = {
			VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
		};
		VkValidationFeaturesEXT validation_features {
			.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
			.enabledValidationFeatureCount = 1,
			.pEnabledValidationFeatures = validation_feature_enables,
		};

		VkInstanceCreateInfo createInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext = &validation_features,
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
	camera_buffer.Destroy(device);
	instance_buffer.Destroy(device);
	for (VulkanTriangleMesh mesh : triangle_meshes) {
		mesh.vertex_buffer.Destroy(device);
		mesh.index_buffer.Destroy(device);
	}
	sbt_buffer.Destroy(device);
	tas.Destroy(*this);
	for (auto& bas: bass) bas.Destroy(*this);
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
		ainfo.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
		VkDescriptorSetLayoutBinding bindings[] = {
			{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			},
			{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			},
			{
				.binding = 2,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			},
			{
				.binding = 3,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			},
		};
		for (auto& b : bindings) {
			b.descriptorCount = 1;
			b.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		}
		VkDescriptorSetLayoutCreateInfo dsl {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)std::size(bindings),
			.pBindings = bindings,
		};
		assert(vkCreateDescriptorSetLayout(device, &dsl, 0, &descriptor_set_layout) == VK_SUCCESS);

		VkPushConstantRange push_range = {
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
			.size = 16,
		};
		VkPipelineLayoutCreateInfo plci {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};
		assert(vkCreatePipelineLayout(device, &plci, 0, &pipeline_layout) == VK_SUCCESS);
	}

	VkShaderModule shader{ 0 };

	LOG("Loading Shaders...");
	{
		shader = load_shader_from_file(device, "../basic.spv");
	}
	LOG("Creating Ray Tracing Pipeline...");
	{
		VkPipelineShaderStageCreateInfo stages[] {
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
				.module = shader,
				.pName = "rgen",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_MISS_BIT_KHR,
				.module = shader,
				.pName = "miss",
			},
			{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
				.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
				.module = shader,
				.pName = "chit",
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

	vkDestroyShaderModule(device, shader, 0);

	LOG("Creating Acceleration Structures...");
	{
		for (const Shape& shape: scene->shapes) {
			add_shape(shape);
		}

		// create instance buffer
		size_t size = instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
		createBuffer(size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, instance_buffer.buffer, instance_buffer.memory);
		VkBuffer instStaging; VkDeviceMemory instStagingMem; createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, instStaging, instStagingMem);
		void* p; vkMapMemory(device, instStagingMem, 0, size, 0, &p); memcpy(p, instances.data(), size); vkUnmapMemory(device, instStagingMem);
		// copy
		{
			VkCommandBuffer cmd = temp_command_buffer();
			VkBufferCopy creg{ 0,0,size }; vkCmdCopyBuffer(cmd, instStaging, instance_buffer, 1, &creg);
			flush_and_destroy_command_buffer(cmd);
		}

		vkDestroyBuffer(device, instStaging, 0);
		vkFreeMemory(device, instStagingMem, 0);

		// Build TLAS
		VkAccelerationStructureGeometryInstancesDataKHR instancesData{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR };
		instancesData.data.deviceAddress = get_buffer_device_address(device, instance_buffer);
		VkAccelerationStructureGeometryKHR iGeom = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
			.geometry = { .instances = instancesData },
		};

		VkAccelerationStructureBuildGeometryInfoKHR tBuildInfo = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
			.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
			.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
			.geometryCount = 1,
			.pGeometries = &iGeom,
		};
		uint32_t maxPrimCountsTLAS[10] = { 10 };
		VkAccelerationStructureBuildSizesInfoKHR tSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tBuildInfo, maxPrimCountsTLAS, &tSizes);

		createBuffer(tSizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tas.buffer.buffer, tas.buffer.memory);
		VkAccelerationStructureCreateInfoKHR tcreate{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR }; tcreate.buffer = tas.buffer.buffer; tcreate.size = tSizes.accelerationStructureSize; tcreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &tcreate, nullptr, &tas.handle);

#if 0
		uint32_t primitive_count = 1;
		VkAccelerationStructureBuildRangeInfoKHR build_range{
			.primitiveCount = primitive_count,
		};
		const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &build_range;

		VkAccelerationStructureBuildSizesInfoKHR sizeInfo{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tBuildInfo, &primitive_count, &sizeInfo);
#endif
		VkAccelerationStructureBuildRangeInfoKHR build_range = {
			.primitiveCount = (uint32_t)instances.size(),
		};
		const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &build_range;

		// scratch buffer
		VkBuffer scratch; VkDeviceMemory scratchMem; createBuffer(tSizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratch, scratchMem);
		VkBufferDeviceAddressInfo saddr{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; saddr.buffer = scratch; VkDeviceAddress scratchAddr = vkGetBufferDeviceAddress(device, &saddr);

		// scratch for TLAS
		VkBuffer tscratch; VkDeviceMemory tscratchMem; createBuffer(tSizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tscratch, tscratchMem);
		saddr.buffer = tscratch; scratchAddr = vkGetBufferDeviceAddress(device, &saddr);

		// build TLAS command
		{
			VkCommandBuffer cmd = temp_command_buffer();
			tBuildInfo.dstAccelerationStructure = tas.handle;
			tBuildInfo.scratchData.deviceAddress = scratchAddr;
			vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tBuildInfo, &pRanges);
			flush_and_destroy_command_buffer(cmd);
		}
		
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
	LOG("Creating Constant Buffers...");
	{
		createBuffer(sizeof(GPU::Camera), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, camera_buffer.buffer, camera_buffer.memory);
		void *data;
		vkMapMemory(device, camera_buffer.memory, 0, VK_WHOLE_SIZE, 0, &data);
		GPU::Camera camera = convert(scene->camera);
		memcpy(data, &camera, sizeof(camera));
		vkUnmapMemory(device, camera_buffer.memory);

		for (int i = 0; i < scene->materials.size(); ++i) {
			std::visit(material_convert_op{material_buffer}, scene->materials[i]);
		}
		material_buffer.Create(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	}
	LOG("Creating Descriptor Set...");
	{
		// Descriptor pool and set
		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              16 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             16 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             16 },
		};
		VkDescriptorPoolCreateInfo dpc{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO }; dpc.maxSets = 1; dpc.poolSizeCount = (uint32_t)std::size(pool_sizes); dpc.pPoolSizes = pool_sizes;
		assert(vkCreateDescriptorPool(device, &dpc, 0, &descriptor_pool) == VK_SUCCESS);
		VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; dsa.descriptorPool = descriptor_pool; dsa.descriptorSetCount = 1; dsa.pSetLayouts = &descriptor_set_layout;
		assert(vkAllocateDescriptorSets(device, &dsa, &descriptor_set) == VK_SUCCESS);

		VkWriteDescriptorSetAccelerationStructureKHR as_info = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
			.accelerationStructureCount = 1,
			.pAccelerationStructures = &tas.handle,
		};

		VkDescriptorImageInfo render_image_info = {
			.imageView = storage_view,
			.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
		};

		VkDescriptorBufferInfo camera_buffer_info = {
			.buffer = camera_buffer,
			.offset = 0,
			.range = sizeof(GPU::Camera),
		};

		VkDescriptorBufferInfo material_buffer_info = {
			.buffer = material_buffer.buffer,
			.offset = 0,
			.range = VK_WHOLE_SIZE,
		};

		VkWriteDescriptorSet writes[] = {
			{
				.pNext = &as_info,
				.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			},
			{
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.pImageInfo = &render_image_info,
			},
			{
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &camera_buffer_info,
			},
			{
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &material_buffer_info,
			},
		};
		for (int i = 0; i < std::size(writes); ++i) {
			writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writes[i].dstSet = descriptor_set;
			writes[i].dstBinding = i;
			writes[i].descriptorCount = 1;
		}
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

	uint32_t frame_count = 0;

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
		vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, 4, &frame_count);

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

		frame_count += 1;
	}
}

void VulkanRawBuffer::Create(GPUDevice& gpu, VkFlags usage)
{
	gpu.createBuffer(data.size(), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, buffer.buffer, buffer.memory);
	void* p;
	vkMapMemory(gpu.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &p);
	memcpy(p, data.data(), data.size());
	vkUnmapMemory(gpu.device, buffer.memory);
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
