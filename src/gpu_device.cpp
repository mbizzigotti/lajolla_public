#include "gpu_device.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#define RGFW_IMGUI_IMPLEMENTATION
#include "3rdparty/imgui_impl_rgfw.h"
#include "3rdparty/imgui_impl_vulkan.h"
#include <fstream>

#ifdef ERROR
#undef ERROR
#endif
#define LOG(...) printf(__VA_ARGS__), putchar('\n')
#define ERROR(...) LOG(__VA_ARGS__), exit(1)
#define LOAD_VULKAN_FUNCTION(NAME) \
	assert(NAME = (PFN_##NAME)vkGetDeviceProcAddr(device, #NAME));

template <typename T>
constexpr bool is_power_of_two(T num) {
	return ((num) & (num - 1)) == 0;
}

template <typename T>
constexpr T align(T current, T alignment) {
	assert(is_power_of_two(alignment));
	return (current + alignment - 1) & ~(alignment - 1);
}

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

VkExtent3D extent3d(VkExtent2D extent2d) { return { extent2d.width, extent2d.height, 1 }; }

VkImageSubresourceLayers default_image_subresource() { return { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }; }

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

VkPipelineShaderStageCreateInfo GPUDevice::load_shader_stage(VkFlags stage, const char* name)
{
	std::string filename = "../" + std::string(name) + ".spv";
	return {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = (VkShaderStageFlagBits)(stage),
		.module = load_shader_from_file(device, filename.c_str()),
		.pName = "main",
	};
}

void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout)
{
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = 0,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	VkPipelineStageFlags srcStage = 0;
	VkPipelineStageFlags dstStage = 0;
	switch (old_layout)
	{
	case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: {
		barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT, srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	} break;
	default: assert(false);
	}
	switch (new_layout)
	{
	case VK_IMAGE_LAYOUT_GENERAL: {
		barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT, dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	} break;
	case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: {
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT, dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} break;
	default: assert(false);
	}
	vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, 0, 0, 0, 1, &barrier);
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

struct shape_convert_op {
	GPU::Shape operator()(const Sphere& shape) {
		return {
			shape.material_id,
			shape.area_light_id,
			shape.interior_medium_id,
			shape.exterior_medium_id,
		};
	}
	GPU::Shape operator()(const TriangleMesh& shape) {
		GPU::Shape result = {
			shape.material_id,
			shape.area_light_id,
			shape.interior_medium_id,
			shape.exterior_medium_id,
			vertex_offset,
			index_offset,
			// (shape.normals.size() > 0) ? 1 : 0, // TODO
		};
		vertex_offset += shape.positions.size();
		index_offset += shape.indices.size();
		return result;
	}
	uint32_t& vertex_offset;
	uint32_t& index_offset;
};

struct light_convert_op {
	GPU::Light operator()(const DiffuseAreaLight& light) {
		assert(light.shape_id >= 0);
		const Shape& shape = scene.shapes[light.shape_id];
		GPU::Light result = {};
		if (std::holds_alternative<TriangleMesh>(shape)) {
			const auto& mesh = std::get<TriangleMesh>(shape);
			result.triangle_sampler = gpu.add_dist_1d(mesh.triangle_sampler);
		}
		result.intensity = light.intensity;
		result.shape_id = light.shape_id;
		result.surface_area = (float)surface_area(shape);
		return result;
	}
	GPU::Light operator()(const Envmap& light) {
		GPU::Light result = {};
		assert(false);
		return result;
	}

	GPUDevice& gpu;
	const Scene& scene;
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
	if (staging_image_buffer.memory) vkUnmapMemory(device, staging_image_buffer.memory);

	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplRgfw_Shutdown();
	ImGui::DestroyContext();

	staging_image_buffer.Destroy(device);
	instance_buffer.buffer.Destroy(device);
	info_buffer.buffer.Destroy(device);
	material_buffer.buffer.Destroy(device);
	shape_buffer.buffer.Destroy(device);
	vertex_buffer.buffer.Destroy(device);
	index_buffer.buffer.Destroy(device);
	uv_buffer.buffer.Destroy(device);
	normal_buffer.buffer.Destroy(device);
	light_buffer.buffer.Destroy(device);
	dist_buffer.buffer.Destroy(device);
	texture_block.Destroy(device);
	scene_block.Destroy(device);
	tonemapper.Destroy(device);

	sbt_buffer.Destroy(device);
	tas.Destroy(*this);
	for (auto& bas: bass) bas.Destroy(*this);
	if (pipeline) vkDestroyPipeline(device, pipeline, 0);
	if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, 0);
	if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, 0);
	if (storage_view) vkDestroyImageView(device, storage_view, 0);
	if (storage_image) vkDestroyImage(device, storage_image, 0);
	if (storage_memory) vkFreeMemory(device, storage_memory, 0);
	if (render_finished_semaphore) vkDestroySemaphore(device, render_finished_semaphore, 0);
	if (image_available_semaphore) vkDestroySemaphore(device, image_available_semaphore, 0);
	if (command_pool) vkDestroyCommandPool(device, command_pool, 0);
	for (int i = 0; i < image_count; ++i) {
		vkDestroyImageView(device, image_views[i], 0);
		vkDestroyFramebuffer(device, frame_buffers[i], 0);
	}
	if (render_pass) vkDestroyRenderPass(device, render_pass, 0);
	if (swap_chain) vkDestroySwapchainKHR(device, swap_chain, 0);
	if (surface) vkDestroySurfaceKHR(instance, surface, 0); 
	if (device) vkDestroyDevice(device, 0);
	if (instance) vkDestroyInstance(instance, 0);
}

void GPUDevice::add_shape_data(const Shape& shape)
{
	const TriangleMesh& mesh = std::get<TriangleMesh>(shape);
	for (const Vector3& position : mesh.positions)
		vertex_buffer.Add(Vector3f(position));
	for (const Vector3i& tri : mesh.indices)
		index_buffer.Add(tri);
	// index_buffer.AddArray(mesh.indices);
	// for (const Vector3& normal : mesh.normals)
	// 	vertex_buffer.Add(Vector3f(normals));
}

void GPUDevice::add_shape(uint32_t index, const GPU::Shape& gpu_shape, const Shape& shape) {
	const TriangleMesh& mesh = std::get<TriangleMesh>(shape);

	VkAccelerationStructureGeometryKHR geometry = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {
			.triangles = {
				.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
				.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
				.vertexData = {.deviceAddress = vertex_buffer.device_address + gpu_shape.vertex_offset * sizeof(Vector3f)},
				.vertexStride = sizeof(Vector3f),
				.maxVertex = (uint32_t)mesh.positions.size(),
				.indexType = VK_INDEX_TYPE_UINT32,
				.indexData = {.deviceAddress = index_buffer.device_address + gpu_shape.index_offset * sizeof(Vector3i)},
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
	VkAccelerationStructureBuildRangeInfoKHR build_range{
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
		.instanceCustomIndex = index,
		.mask = 0xFF,
		.instanceShaderBindingTableRecordOffset = 0,
		.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
		.accelerationStructureReference = bas.address,
	};

	bass.emplace_back(bas);
	instance_buffer.Add(asInstance);
}

GPU::TableDist1D GPUDevice::add_dist_1d(const TableDist1D& table)
{
	uint32_t pmf_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.pmf);

	uint32_t cdf_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.cdf);

	return GPU::TableDist1D{ uint4(
		pmf_offset, (uint32_t)(table.pmf.size()),
		cdf_offset, (uint32_t)(table.cdf.size())
	) };
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
		VkPhysicalDeviceRobustness2FeaturesEXT           robustness_features             { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT };

		device_features                 .pNext = &device_address_features;
		device_address_features         .pNext = &acceleration_structure_features;
		acceleration_structure_features .pNext = &ray_tracing_features;
		ray_tracing_features            .pNext = &robustness_features;

		auto vkGetPhysicalDeviceFeatures2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2");
		assert(vkGetPhysicalDeviceFeatures2);
		vkGetPhysicalDeviceFeatures2(physical_device, &device_features);

		// request enabling of required features (if supported)
		device_address_features.bufferDeviceAddress = VK_TRUE;
		acceleration_structure_features.accelerationStructure = VK_TRUE;
		ray_tracing_features.rayTracingPipeline = VK_TRUE;
		robustness_features.nullDescriptor = VK_TRUE;

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
			VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
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
			.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
			.preTransform = capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR,
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
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.extent = extent3d(image_extent),
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.tiling = VK_IMAGE_TILING_OPTIMAL,
			.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
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
		siv.image = storage_image; siv.viewType = VK_IMAGE_VIEW_TYPE_2D; siv.format = VK_FORMAT_R32G32B32A32_SFLOAT;
		siv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; siv.subresourceRange.baseMipLevel = 0; siv.subresourceRange.levelCount = 1;
		siv.subresourceRange.baseArrayLayer = 0; siv. subresourceRange.layerCount = 1;
		vkCreateImageView(device, &siv, nullptr, &storage_view);
	}
	LOG("Creating Layouts...");
	{
		texture_block.add_binding("sampler",  VK_DESCRIPTOR_TYPE_SAMPLER);
		texture_block.add_binding("textures", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);

		texture_block.shader_stages = VK_SHADER_STAGE_RAYGEN_BIT_KHR
			                        | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		texture_block.CreateLayout(device);

		scene_block.add_binding("image",    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
		scene_block.add_binding("as",       VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
		scene_block.add_binding("info",     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		scene_block.add_binding("mat",      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("shape",    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("position", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("triangle", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("uv",       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("normal",   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("light",    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("tex",      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("dist",     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

		scene_block.shader_stages = VK_SHADER_STAGE_RAYGEN_BIT_KHR
			                      | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
		scene_block.CreateLayout(device);

		VkPushConstantRange push_range = {
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
			.size = sizeof(GPU::PerFrameInfo),
		};
		VkDescriptorSetLayout layouts[] = {
			texture_block.descriptor_set_layout,
			scene_block.descriptor_set_layout,
		};
		VkPipelineLayoutCreateInfo plci {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = (uint32_t)std::size(layouts),
			.pSetLayouts = layouts,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};
		assert(vkCreatePipelineLayout(device, &plci, 0, &pipeline_layout) == VK_SUCCESS);
	}

	VkPipelineShaderStageCreateInfo stages[4] = {};

	LOG("Loading Shaders...");
	{
		stages[0] = load_shader_stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rgen_path_tracing");
		stages[1] = load_shader_stage(VK_SHADER_STAGE_MISS_BIT_KHR, "miss");
		stages[2] = load_shader_stage(VK_SHADER_STAGE_MISS_BIT_KHR, "miss_shadow");
		stages[3] = load_shader_stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "chit_tri");
	//	stages[4] = load_shader_stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, "ahit");
	}
	LOG("Creating Ray Tracing Pipeline...");
	{
		// Shader groups: raygen(0), missprimary(1), missshadow(2), hitgroup(3), shadowgroup(4)
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
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
				.generalShader      = 2,
				.closestHitShader   = VK_SHADER_UNUSED_KHR,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			},
			{
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
				.generalShader      = VK_SHADER_UNUSED_KHR,
				.closestHitShader   = 3,
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

	for (const auto& stage: stages)
		if (stage.module) vkDestroyShaderModule(device, stage.module, 0);

	LOG("Creating Constant Buffers...");
	{
		GPU::SceneInfo scene_info = {};
		{
			for (const Material& material : scene->materials) {
				std::visit(material_convert_op{ material_buffer }, material);
			}
			material_buffer.Create(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		}
		{
			uint32_t vertex_offset = 0;
			uint32_t index_offset = 0;
			for (const Shape& shape : scene->shapes) {
				shape_buffer.Add(std::visit(shape_convert_op{ vertex_offset, index_offset }, shape));
			}
			shape_buffer.Create(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		}
		{
			for (const Shape& shape : scene->shapes) {
				add_shape_data(shape);
			}
			VkFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
						  | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
						  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
			vertex_buffer.CreateFromStaging(*this, usage);
			vertex_buffer.GetDeviceAddress(device);
			index_buffer.CreateFromStaging(*this, usage);
			index_buffer.GetDeviceAddress(device);

			uv_buffer.CreateFromStaging(*this, usage);
			normal_buffer.CreateFromStaging(*this, usage);
		}
		{
			uint32_t vertex_offset = 0, index_offset = 0;
			for (uint32_t i = 0; i < scene->shapes.size(); ++i) {
				const Shape& shape = scene->shapes[i];
				GPU::Shape gpu_shape = std::visit(shape_convert_op{ vertex_offset, index_offset }, shape);
				add_shape(i, gpu_shape, shape);
			}
			VkFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
						  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
			instance_buffer.CreateFromStaging(*this, usage);
			instance_buffer.GetDeviceAddress(device);
		}
		{
			for (const Light& light : scene->lights) {
				light_buffer.Add(std::visit(light_convert_op{*this, *scene}, light));
			}
			light_buffer.CreateFromStaging(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		}
		{
			scene_info.light_dist = add_dist_1d(scene->light_dist);
			dist_buffer.CreateFromStaging(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		}
		{
			scene_info.bounds.center = scene->bounds.center;
			scene_info.bounds.radius = scene->bounds.radius;
			scene_info.camera = convert(scene->camera);
			info_buffer.Add(scene_info);
			info_buffer.CreateFromStaging(*this, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		}
	}
	LOG("Creating Acceleration Structures...");
	{
		VkAccelerationStructureGeometryInstancesDataKHR instancesData = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
			.data = {.deviceAddress = instance_buffer.device_address},
		};
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
		uint32_t maxPrimCountsTLAS[10] = { 1000 };
		VkAccelerationStructureBuildSizesInfoKHR tSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tBuildInfo, maxPrimCountsTLAS, &tSizes);

		createBuffer(tSizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tas.buffer.buffer, tas.buffer.memory);
		VkAccelerationStructureCreateInfoKHR tcreate{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR }; tcreate.buffer = tas.buffer.buffer; tcreate.size = tSizes.accelerationStructureSize; tcreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &tcreate, nullptr, &tas.handle);

		VkAccelerationStructureBuildRangeInfoKHR build_range = {
			.primitiveCount = (uint32_t)(scene->shapes.size()),
		};
		const VkAccelerationStructureBuildRangeInfoKHR* pRanges = &build_range;

		// scratch for TLAS
		VkBuffer tscratch; VkDeviceMemory tscratchMem;
		createBuffer(tSizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tscratch, tscratchMem);

		// build TLAS command
		{
			VkCommandBuffer cmd = temp_command_buffer();
			tBuildInfo.dstAccelerationStructure = tas.handle;
			tBuildInfo.scratchData.deviceAddress = get_buffer_device_address(device, tscratch);
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
		uint32_t handleSizeAligned = align(rtprops.shaderGroupHandleSize, rtprops.shaderGroupHandleAlignment);
		uint32_t baseAlignment = rtprops.shaderGroupBaseAlignment;

		uint32_t groupCount = 4;
		std::vector<char> shaderHandleStorage(groupCount * handleSize);
		vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount, shaderHandleStorage.size(), shaderHandleStorage.data());

		// create SBT buffer (host visible)
		VkDeviceSize sbtSize = groupCount * baseAlignment;
		createBuffer(sbtSize, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sbt_buffer.buffer, sbt_buffer.memory);
		void* sbtMap;
		vkMapMemory(device, sbt_buffer.memory, 0, sbtSize, 0, &sbtMap);
		for (uint32_t i = 0; i < groupCount; ++i)
			memcpy(reinterpret_cast<char*>(sbtMap) + i * baseAlignment, shaderHandleStorage.data() + i * handleSize, handleSize);
		vkUnmapMemory(device, sbt_buffer.memory);
		VkBufferDeviceAddressInfo sbtAddrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO }; sbtAddrInfo.buffer = sbt_buffer; VkDeviceAddress sbtAddr = vkGetBufferDeviceAddress(device, &sbtAddrInfo);

		rgen_sbt = { .deviceAddress = sbtAddr + 0 * baseAlignment, .stride = baseAlignment, .size = baseAlignment };
		miss_sbt = { .deviceAddress = sbtAddr + 1 * baseAlignment, .stride = baseAlignment, .size = 2 * baseAlignment };
		chit_sbt = { .deviceAddress = sbtAddr + 3 * baseAlignment, .stride = baseAlignment, .size = baseAlignment };
	}
	LOG("Creating Descriptor Sets...");
	{
		// Descriptor pool and set
		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,     64 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,              64 },
			{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              16 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             16 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             16 },
			{ VK_DESCRIPTOR_TYPE_SAMPLER,                     1 },
		};
		VkDescriptorPoolCreateInfo pool_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
			.maxSets = 16,
			.poolSizeCount = (uint32_t)std::size(pool_sizes),
			.pPoolSizes = pool_sizes,
		};
		assert(vkCreateDescriptorPool(device, &pool_info, 0, &descriptor_pool) == VK_SUCCESS);

		scene_block.Allocate(device, descriptor_pool);
		texture_block.Allocate(device, descriptor_pool);

		VkWriteDescriptorSet writes[] = {
			scene_block.write("image",    storage_view),
			scene_block.write("as",       &tas.handle),
			scene_block.write("info",     info_buffer),
			scene_block.write("mat",	  material_buffer),
			scene_block.write("shape",    shape_buffer),
			scene_block.write("position", vertex_buffer),
			scene_block.write("triangle", index_buffer),
			scene_block.write("uv",       uv_buffer),
			scene_block.write("normal",   normal_buffer),
		    scene_block.write("light",    light_buffer),
		//  scene_block.write("tex",      texture_buffer),
		    scene_block.write("dist",     dist_buffer),
		};
		vkUpdateDescriptorSets(device, (uint32_t)std::size(writes), writes, 0, 0);
	}
	LOG("Creating Render Pass...");
	{
		VkAttachmentDescription attachment = {};
		attachment.format = swap_format;
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachment.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
		attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		VkAttachmentReference color_attachment = {};
		color_attachment.attachment = 0;
		color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment;
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		VkRenderPassCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		info.attachmentCount = 1;
		info.pAttachments = &attachment;
		info.subpassCount = 1;
		info.pSubpasses = &subpass;
		info.dependencyCount = 1;
		info.pDependencies = &dependency;
		assert(vkCreateRenderPass(device, &info, 0, &render_pass) == VK_SUCCESS);
	}
	LOG("Creating Framebuffers...");
	{
		VkImageView attachment[1];
		VkFramebufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		info.renderPass = render_pass;
		info.attachmentCount = 1;
		info.pAttachments = attachment;
		info.width = image_extent.width;
		info.height = image_extent.height;
		info.layers = 1;
		for (uint32_t i = 0; i < image_count; i++)
		{
			attachment[0] = image_views[i];
			assert(vkCreateFramebuffer(device, &info, 0, &frame_buffers[i]) == VK_SUCCESS);
		}
	}
	LOG("Creating Staging Image Buffer...");
	{
		VkDeviceSize size = image_extent.width * image_extent.height * 4 * sizeof(float);
		VkFlags memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory_flags, staging_image_buffer.buffer, staging_image_buffer.memory);
		vkMapMemory(device, staging_image_buffer.memory, 0, VK_WHOLE_SIZE, 0, (void**)&storage_mapped);
	}
	LOG("Setting up ImGui...");
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

		// Setup Platform/Renderer backends
		ImGui_ImplRgfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = instance;
		init_info.PhysicalDevice = physical_device;
		init_info.Device = device;
		init_info.QueueFamily = 0;
		init_info.Queue = graphics_queue;
		init_info.DescriptorPool = descriptor_pool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = image_count;
		init_info.PipelineInfoMain.RenderPass = render_pass;
		init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		init_info.CheckVkResultFn = [](VkResult err) {
			if (err == VK_SUCCESS) return;
			fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
			if (err < 0) abort();
		};
		ImGui_ImplVulkan_Init(&init_info);
	}
	LOG("Creating Tonemapper...");
	{
		tonemapper.Create(*this);
	}
	target_sample_count = (uint32_t)(scene->options.samples_per_pixel);
}

void GPUDevice::render(RGFW_window* window)
{
	VkCommandBufferAllocateInfo ca{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; ca.commandPool = command_pool; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
	VkCommandBuffer cmd; vkAllocateCommandBuffers(device, &ca, &cmd);

	uint32_t width = image_extent.width;
	uint32_t height = image_extent.height;
	bool show_controls = false;
	bool show_window = true;

	uint32_t frame_count = 0;

	while (!RGFW_window_shouldClose(window))
	{
		RGFW_pollEvents();

		if (RGFW_isKeyPressed(RGFW_r))
			frame_count = 0;
		if (RGFW_isKeyPressed(RGFW_escape))
			show_window = !show_window;
		if (RGFW_isKeyPressed(RGFW_h))
			show_controls = !show_controls;

		bool want_save = RGFW_isKeyPressed(RGFW_s) || (frame_count + 1 == target_sample_count);

		uint32_t image_index;
		vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX, image_available_semaphore, 0, &image_index);

		tonemapper.SetImages(device, storage_view, image_views[image_index]);

		VkCommandBufferBeginInfo bi2{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &bi2);

		// transition storage image to general
		VkImageMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.image = storage_image; barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; barrier.subresourceRange.baseMipLevel = 0; barrier.subresourceRange.levelCount = 1; barrier.subresourceRange.baseArrayLayer = 0; barrier.subresourceRange.layerCount = 1; barrier.srcAccessMask = 0; barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		// bind pipeline and descriptor sets and trace
		VkDescriptorSet sets[] = {
			texture_block.descriptor_set,
			scene_block.descriptor_set,
		};
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline_layout, 0, std::size(sets), sets, 0, 0);
		vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 0, 4, &frame_count);

		VkStridedDeviceAddressRegionKHR callable_sbt{};
		vkCmdTraceRaysKHR(cmd, &rgen_sbt, &miss_sbt, &chit_sbt, &callable_sbt, width, height, 1);

		// transition storage image to src
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.image = storage_image; barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; barrier.subresourceRange.baseMipLevel = 0; barrier.subresourceRange.levelCount = 1; barrier.subresourceRange.baseArrayLayer = 0; barrier.subresourceRange.layerCount = 1; barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		// transition swapchain image to transfer dst
		VkImage dst = images[image_index];
		VkImageMemoryBarrier toCopy{}; toCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; toCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; toCopy.newLayout = VK_IMAGE_LAYOUT_GENERAL; toCopy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toCopy.image = dst; toCopy.subresourceRange = barrier.subresourceRange; toCopy.srcAccessMask = 0; toCopy.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toCopy);

		tonemapper.Tonemap(cmd, width, height);

		if (want_save)
		{
			transition_image(cmd, storage_image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			VkBufferImageCopy region = {
				.imageSubresource = default_image_subresource(),
				.imageExtent = extent3d(image_extent),
			};
			vkCmdCopyImageToBuffer(cmd, storage_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_image_buffer, 1, &region);
		}

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplRgfw_NewFrame();
		ImGui::NewFrame();
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
		if (show_window)
		{
			ImGui::Begin("Debug", 0, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize);
			{
				ImGui::Text("Sample Count: %u", frame_count);
				if (show_controls)
				{
					ImGui::SliderFloat("Exposure", &tonemapper.exposure, 0.01f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
				}
			}
			ImGui::End();
		}
		ImGui::Render();
		{
			VkRenderPassBeginInfo info = {
				.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
				.renderPass = render_pass,
				.framebuffer = frame_buffers[image_index],
				.renderArea = {.extent = image_extent},
			};
			vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
		}
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);

		VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; VkSemaphore waitSem[] = { image_available_semaphore }; VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_TRANSFER_BIT };
		submit.waitSemaphoreCount = 1; submit.pWaitSemaphores = waitSem; submit.pWaitDstStageMask = waitStages; submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd; VkSemaphore sig[] = { render_finished_semaphore }; submit.signalSemaphoreCount = 1; submit.pSignalSemaphores = sig;
		vkQueueSubmit(graphics_queue, 1, &submit, VK_NULL_HANDLE);

		VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR }; pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = sig; pi.swapchainCount = 1; pi.pSwapchains = &swap_chain; pi.pImageIndices = &image_index;
		vkQueuePresentKHR(present_queue, &pi);
		vkQueueWaitIdle(present_queue);

		if (want_save)
		{
			int width = (int)(image_extent.width);
			int height = (int)(image_extent.height);
			char filename[256] = {};
			snprintf(filename, 256, "image_%uspp.exr", frame_count + 1);
			imwrite_raw(filename, storage_mapped, width, height);
			LOG("Saved to \"%s\"", filename);
		}

		frame_count += 1;
	}

	vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

void VulkanRawBuffer::Create(GPUDevice& gpu, VkFlags usage)
{
	gpu.createBuffer(data.size(), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, buffer.buffer, buffer.memory);
	void* p;
	vkMapMemory(gpu.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &p);
	memcpy(p, data.data(), data.size());
	vkUnmapMemory(gpu.device, buffer.memory);
}

void VulkanRawBuffer::CreateFromStaging(GPUDevice& gpu, VkFlags usage)
{
	if (data.size() <= 0) return;

	usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT; // required
	gpu.createBuffer(data.size(), usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer.buffer, buffer.memory);

	VulkanBuffer staging{};
	gpu.createBuffer(data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging.buffer, staging.memory);
	void* p;
	vkMapMemory(gpu.device, staging.memory, 0, data.size(), 0, &p);
	memcpy(p, data.data(), data.size());
	vkUnmapMemory(gpu.device, staging.memory);

	VkCommandBuffer cmd = gpu.temp_command_buffer();
	VkBufferCopy copyRegion{ 0, 0, data.size() };
	vkCmdCopyBuffer(cmd, staging, buffer, 1, &copyRegion);
	gpu.flush_and_destroy_command_buffer(cmd);

	staging.Destroy(gpu.device);
}

void VulkanAccelerationStructure::Destroy(struct GPUDevice& gpu)
{
	if (handle) gpu.vkDestroyAccelerationStructureKHR(gpu.device, handle, 0);
	buffer.Destroy(gpu.device);
}

void VulkanRawBuffer::GetDeviceAddress(VkDevice device)
{
	device_address = get_buffer_device_address(device, buffer);
}

void VulkanBuffer::Destroy(VkDevice device)
{
	if (buffer) vkDestroyBuffer(device, buffer, 0);
	if (memory) vkFreeMemory(device, memory, 0);
}

void Tonemapper::Create(GPUDevice& gpu)
{
	block.add_binding("in", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
	block.add_binding("out", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	block.shader_stages = VK_SHADER_STAGE_COMPUTE_BIT;
	block.CreateLayout(gpu.device);
	block.Allocate(gpu.device, gpu.descriptor_pool);

	VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.size = sizeof(float),
	};
	VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &block.descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};
	assert(vkCreatePipelineLayout(gpu.device, &pipeline_layout_info, 0, &pipeline_layout) == VK_SUCCESS);

	VkComputePipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = gpu.load_shader_stage(VK_SHADER_STAGE_COMPUTE_BIT, "tonemap"),
		.layout = pipeline_layout,
	};
	assert(vkCreateComputePipelines(gpu.device, 0, 1, &pipeline_info, 0, &pipeline) == VK_SUCCESS);
	vkDestroyShaderModule(gpu.device, pipeline_info.stage.module, 0);
}

void Tonemapper::Destroy(VkDevice device)
{
	block.Destroy(device);
	if (pipeline) vkDestroyPipeline(device, pipeline, 0);
	if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, 0);
}

void Tonemapper::Tonemap(VkCommandBuffer cmd, uint32_t width, uint32_t height)
{
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &block.descriptor_set, 0, 0);
	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &exposure);
	vkCmdDispatch(cmd, align(width, 8u), align(height, 8u), 1);
}

// ImGui does not recommend putting itself in a DLL,
// so I guess this will have to stay here...
#include "3rdparty/imgui.cpp"
#include "3rdparty/imgui_demo.cpp"
#include "3rdparty/imgui_draw.cpp"
#include "3rdparty/imgui_tables.cpp"
#include "3rdparty/imgui_widgets.cpp"
#include "3rdparty/imgui_impl_vulkan.cpp"
