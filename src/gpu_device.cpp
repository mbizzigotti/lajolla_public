#include "gpu_device.h"
#define IMGUI_DEFINE_MATH_OPERATORS
#define RGFW_IMGUI_IMPLEMENTATION
#include "3rdparty/imgui_impl_rgfw.h"
#include "3rdparty/imgui_impl_vulkan.h"
#include "timer.h"
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

template <typename T>
constexpr T ceil_div(T num, T den) {
	return (num + den - 1) / den;
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

VkWriteDescriptorSet ShaderParameterBlock::write(const char* name, VkAccelerationStructureKHR* as)
{
	VkWriteDescriptorSetAccelerationStructureKHR as_info = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		.accelerationStructureCount = 1,
		.pAccelerationStructures = as,
	};
	assert(descriptor_map.contains(name));
	uint32_t binding = descriptor_map[name];
	assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
	return {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = push_write_structure(as_info),
		.dstSet = descriptor_set,
		.dstBinding = binding,
		.descriptorCount = descriptors[binding].count,
		.descriptorType = descriptors[binding].type,
	};
}

VkWriteDescriptorSet ShaderParameterBlock::write(const char* name, VkImageView image_view, VkImageLayout layout)
{
	VkDescriptorImageInfo render_image_info = {
		.imageView = image_view,
		.imageLayout = layout,
	};
	assert(descriptor_map.contains(name));
	uint32_t binding = descriptor_map[name];
	assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
		|| descriptors[binding].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	return {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = binding,
		.descriptorCount = descriptors[binding].count,
		.descriptorType = descriptors[binding].type,
		.pImageInfo = push_write_structure(render_image_info),
	};
}

VkWriteDescriptorSet ShaderParameterBlock::write(const char* name, VkBuffer buffer)
{
	VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};
	assert(descriptor_map.contains(name));
	uint32_t binding = descriptor_map[name];
	assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
		|| descriptors[binding].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	return {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = binding,
		.descriptorCount = descriptors[binding].count,
		.descriptorType = descriptors[binding].type,
		.pBufferInfo = push_write_structure(buffer_info),
	};
}

VkWriteDescriptorSet ShaderParameterBlock::write(const char* name, VkSampler sampler)
{
	VkDescriptorImageInfo image_info = {
		.sampler = sampler,
	};
	assert(descriptor_map.contains(name));
	uint32_t binding = descriptor_map[name];
	assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_SAMPLER);
	return {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = binding,
		.descriptorCount = descriptors[binding].count,
		.descriptorType = descriptors[binding].type,
		.pImageInfo = push_write_structure(image_info),
	};
}

VkWriteDescriptorSet ShaderParameterBlock::write_many(const char* name, VkDescriptorImageInfo* images, uint32_t count)
{
	assert(descriptor_map.contains(name));
	uint32_t binding = descriptor_map[name];
	assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
		|| descriptors[binding].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
	return {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set,
		.dstBinding = binding,
		.descriptorCount = count,
		.descriptorType = descriptors[binding].type,
		.pImageInfo = images,
	};
}

struct filter_convert_op {
	GPU::Filter operator()(const Box& filter) const { return { GPU::Box, static_cast<float>(filter.width) }; }
	GPU::Filter operator()(const Tent& filter) const { return { GPU::Tent, static_cast<float>(filter.width) }; }
	GPU::Filter operator()(const Gaussian& filter) const { return { GPU::Gaussian, static_cast<float>(filter.stddev) }; }
};

GPU::Camera convert(const Camera &camera)
{
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

struct texture_spectrum_convert_op {
	GPU::Parameter operator()(const ConstantTexture<Spectrum>& texture) {
		GPU::Parameter result = {};
		result.value = Vector3f(texture.value);
		result.texture_id = -1;
		return result;
	}
	GPU::Parameter operator()(const ImageTexture<Spectrum>& texture) {
		GPU::Parameter result = {};
		result.texture_id = texture_infos.Count<GPU::TextureInfo>();
		GPU::TextureInfo info = {};
		info.index = texture.texture_id;
		info.offset = { texture.uoffset, texture.voffset };
		info.scale = { texture.uscale, texture.vscale };
		texture_infos.Add(info);
		return result;
	}
	GPU::Parameter operator()(const CheckerboardTexture<Spectrum>& texture) {
		GPU::Parameter result = {};
		result.texture_id = texture_infos.Count<GPU::TextureInfo>();
		GPU::TextureInfo info = {};
		info.color0 = texture.color0;
		info.color1 = texture.color1;
		info.offset = { texture.uoffset, texture.voffset };
		info.scale = { texture.uscale, texture.vscale };
		info.is_checkerboard = 1;
		texture_infos.Add(info);
		return result;
	}

	VulkanRawBuffer& texture_infos;
};

struct texture_real_convert_op {
	GPU::Parameter operator()(const ConstantTexture<Real>& texture) {
		GPU::Parameter result = {};
		result.value.x = (float)(texture.value);
		result.texture_id = -1;
		return result;
	}
	GPU::Parameter operator()(const ImageTexture<Real>& texture) {
		GPU::Parameter result = {};
		result.texture_id = texture_infos.Count<GPU::TextureInfo>();
		GPU::TextureInfo info = {};
		info.index = texture.texture_id;
		info.offset = { texture.uoffset, texture.voffset };
		info.scale = { texture.uscale, texture.vscale };
		texture_infos.Add(info);
		return result;
	}
	GPU::Parameter operator()(const CheckerboardTexture<Real>& texture) {
		GPU::Parameter result = {};
		result.texture_id = texture_infos.Count<GPU::TextureInfo>();
		GPU::TextureInfo info = {};
		info.color0.x = texture.color0;
		info.color1.x = texture.color1;
		info.offset = { texture.uoffset, texture.voffset };
		info.scale = { texture.uscale, texture.vscale };
		info.is_checkerboard = 1;
		texture_infos.Add(info);
		return result;
	}

	VulkanRawBuffer& texture_infos;
};

struct material_convert_op {
	void operator()(const Lambertian& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::Lambertian;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.reflectance);
		raw.Add(material);
	}
	void operator()(const RoughPlastic& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::RoughPlastic;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.diffuse_reflectance);
		material.param1 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.specular_reflectance);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		material.eta = bsdf.eta;
		raw.Add(material);
	}
	void operator()(const RoughDielectric& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::RoughDielectric;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.specular_transmittance);
		material.param1 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.specular_reflectance);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		material.eta = bsdf.eta;
		raw.Add(material);
	}
	void operator()(const DisneyDiffuse& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneyDiffuse;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.base_color);
		material.param1 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.subsurface);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		raw.Add(material);
	}
	void operator()(const DisneyMetal& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneyMetal;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.base_color);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		material.param3 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.anisotropic);
		raw.Add(material);
	}
	void operator()(const DisneyGlass& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneyGlass;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.base_color);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		material.param3 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.anisotropic);
		material.eta = bsdf.eta;
		raw.Add(material);
	}
	void operator()(const DisneyClearcoat& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneyClearcoat;
		material.param4 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.clearcoat_gloss);
		raw.Add(material);
	}
	void operator()(const DisneySheen& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneySheen;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.base_color);
		material.param5 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.sheen_tint);
		raw.Add(material);
	}
	void operator()(const DisneyBSDF& bsdf) {
		GPU::Material material = {};
		material.type = GPU::MaterialType::DisneyBSDF;
		material.param0 = std::visit(texture_spectrum_convert_op{ texture_buffer }, bsdf.base_color);
		material.param1 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.subsurface);
		material.param2 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.roughness);
		material.param3 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.anisotropic);
		material.param4 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.clearcoat_gloss);
		material.param5 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.sheen_tint);
		material.param6 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.specular_transmission);
		material.param7 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.metallic);
		material.param8 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.specular);
		material.param9 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.specular_tint);
		material.param10 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.sheen);
		material.param11 = std::visit(texture_real_convert_op{ texture_buffer }, bsdf.clearcoat);
		material.eta = bsdf.eta;
		raw.Add(material);
	}

	VulkanRawBuffer& raw;
	VulkanRawBuffer& texture_buffer;
};

struct shape_convert_op {
	GPU::Shape operator()(const Sphere& shape) {
		GPU::Shape result = {};
		result.material_id = shape.material_id;
		result.area_light_id = shape.area_light_id;
		result.interior_medium_id = shape.interior_medium_id;
		result.exterior_medium_id = shape.exterior_medium_id;
		result.position = shape.position;
		result.radius = shape.radius;
		result.flags = GPU::SHAPE_IS_SPHERE;
		return result;
	}
	GPU::Shape operator()(const TriangleMesh& shape) {
		GPU::Shape result = {};
		result.material_id = shape.material_id;
		result.area_light_id = shape.area_light_id;
		result.interior_medium_id = shape.interior_medium_id;
		result.exterior_medium_id = shape.exterior_medium_id;
		result.vertex_offset = vertex_offset;
		result.index_offset = index_offset;
		if (shape.uvs    .size() > 0) result.flags |= GPU::SHAPE_HAS_UVS;
		if (shape.normals.size() > 0) result.flags |= GPU::SHAPE_HAS_NORMALS;
		vertex_offset += shape.positions.size();
		index_offset += shape.indices.size();
		return result;
	}
	uint32_t& vertex_offset;
	uint32_t& index_offset;
};

struct get_shape_geometry_op {
	VkAccelerationStructureGeometryKHR operator()(const Sphere& shape) {
		return {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR,
			.geometry = {
				.aabbs = {
					.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
					.data = {.deviceAddress = aabb_buffer.device_address + sphere_index * sizeof(VkAabbPositionsKHR)},
					.stride = sizeof(VkAabbPositionsKHR),
				}
			},
			.flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
		};
	}
	VkAccelerationStructureGeometryKHR operator()(const TriangleMesh& shape) {
		return {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
			.geometry = {
				.triangles = {
					.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
					.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
					.vertexData = {.deviceAddress = vertex_buffer.device_address + gpu_shape.vertex_offset * sizeof(Vector3f)},
					.vertexStride = sizeof(Vector3f),
					.maxVertex = (uint32_t)shape.positions.size(),
					.indexType = VK_INDEX_TYPE_UINT32,
					.indexData = {.deviceAddress = index_buffer.device_address + gpu_shape.index_offset * sizeof(Vector3i)},
				}
			},
			.flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
		};
	}

	const VulkanRawBuffer &vertex_buffer;
	const VulkanRawBuffer &index_buffer;
	const VulkanRawBuffer &aabb_buffer;
	const GPU::Shape &gpu_shape;
	const uint32_t &sphere_index;
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
		result.shape_id = -1;
		return result;
	}

	GPUDevice& gpu;
	const Scene& scene;
};

struct phase_function_convert_op {
	GPU::PhaseFunction operator()(const IsotropicPhase& p) const {
		GPU::PhaseFunction result = {};
		result.type = GPU::PhaseFunctionType::Isotropic;
		return result;
	}
	GPU::PhaseFunction operator()(const HenyeyGreenstein& p) const {
		GPU::PhaseFunction result = {};
		result.type = GPU::PhaseFunctionType::HenyeyGreenstein;
		result.param.x = p.g;
		return result;
	}
};

template <typename T>
struct volume_convert_op {
	GPU::Volume operator()(const ConstantVolume<T>& volume) const {
		GPU::Volume result = {};
		result.index = -1;
		result.max_data = Vector3f(volume.value);
		return result;
	}
	GPU::Volume operator()(const GridVolume<T>& volume) const {
		GPU::Volume result = {};
		result.max_data = Vector3f(volume.max_data * volume.scale);
		result.index = volume_id;
		result.p_min = Vector3f(volume.p_min);
		result.p_max = Vector3f(volume.p_max);
		result.resolution = volume.resolution;
		result.scale = (float)(volume.scale);
		return result;
	}
	int volume_id;
};

struct medium_convert_op {
	GPU::Medium operator()(const HomogeneousMedium& m) const {
		GPU::Medium result = {};
		result.albedo  = GPU::Volume{ .max_data = Vector3f(m.sigma_a) };
		result.density = GPU::Volume{ .max_data = Vector3f(m.sigma_s) };
		result.phase_function = std::visit(phase_function_convert_op{}, m.phase_function);
		return result;
	}
	GPU::Medium operator()(const HeterogeneousMedium& m) const {
		GPU::Medium result;
		result.albedo = std::visit(volume_convert_op<Spectrum>{m.albedo_volume_id}, m.albedo);
		result.density = std::visit(volume_convert_op<Spectrum>{m.density_volume_id}, m.density);
		result.phase_function = std::visit(phase_function_convert_op{}, m.phase_function);
		result.is_heterogenious = 1;
		return result;
	}
};

GPU::EnvironmentMap convert(GPUDevice& gpu, const Envmap& envmap)
{
	GPU::EnvironmentMap result = {};
	result.to_local = (Matrix4x4f)(envmap.to_local);
	result.to_world = (Matrix4x4f)(envmap.to_world);
	result.sampling_dist = gpu.add_dist_2d(envmap.sampling_dist);
	result.scale = (float)(envmap.scale);
	result.values = std::visit(texture_spectrum_convert_op{gpu.texture_buffer}, envmap.values);
	return result;
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
			.pNext = __DEBUG__ ? &validation_features : 0,
			.pApplicationInfo = &application_info,
			.enabledLayerCount = __DEBUG__ ? (uint32_t)std::size(layers) : 0,
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

void GPUDevice::add_texture(const Mipmap3& mipmap)
{
	VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
	VulkanImage texture{};
	VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = { (uint32_t)(mipmap.images[0].width), (uint32_t)(mipmap.images[0].height), 1 },
		.mipLevels = (uint32_t)(mipmap.images.size()),
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	assert(vkCreateImage(device, &image_info, nullptr, &texture.image) == VK_SUCCESS);

	VkMemoryRequirements memReq;
	vkGetImageMemoryRequirements(device, texture.image, &memReq);

	VkMemoryAllocateInfo ainfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	ainfo.allocationSize = memReq.size;
	ainfo.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(device, &ainfo, nullptr, &texture.memory);
	vkBindImageMemory(device, texture.image, texture.memory, 0);

	VkImageViewCreateInfo siv{}; siv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	siv.image = texture.image; siv.viewType = VK_IMAGE_VIEW_TYPE_2D; siv.format = format;
	siv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; siv.subresourceRange.baseMipLevel = 0;
	siv.subresourceRange.levelCount = image_info.mipLevels;
	siv.subresourceRange.baseArrayLayer = 0; siv.subresourceRange.layerCount = 1;
	vkCreateImageView(device, &siv, nullptr, &texture.view);

	for (int mip = 0; mip < mipmap.images.size(); ++mip)
	{
		write_image(texture.image, mipmap.images[mip], format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mip);
	}

	textures.emplace_back(texture);
}

int GPUDevice::add_volume(const GridVolume<Spectrum>& volume)
{
	VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;
	VulkanImage texture{};
	VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_3D,
		.format = format,
		.extent = {
			(uint32_t)(volume.resolution.x),
			(uint32_t)(volume.resolution.y),
			(uint32_t)(volume.resolution.z)
		},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	assert(vkCreateImage(device, &image_info, nullptr, &texture.image) == VK_SUCCESS);

	VkMemoryRequirements memReq;
	vkGetImageMemoryRequirements(device, texture.image, &memReq);

	VkMemoryAllocateInfo ainfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	ainfo.allocationSize = memReq.size;
	ainfo.memoryTypeIndex = find_memory_type(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	vkAllocateMemory(device, &ainfo, nullptr, &texture.memory);
	vkBindImageMemory(device, texture.image, texture.memory, 0);

	VkImageViewCreateInfo siv{}; siv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	siv.image = texture.image; siv.viewType = VK_IMAGE_VIEW_TYPE_3D; siv.format = format;
	siv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; siv.subresourceRange.baseMipLevel = 0;
	siv.subresourceRange.levelCount = image_info.mipLevels;
	siv.subresourceRange.baseArrayLayer = 0; siv.subresourceRange.layerCount = 1;
	vkCreateImageView(device, &siv, nullptr, &texture.view);

	write_volume(texture.image, volume, format, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	int index = (int)std::size(volumes);
	volumes.emplace_back(texture);
	return index;
}

void GPUDevice::add_shape_data(const Shape& shape)
{
	if (std::holds_alternative<TriangleMesh>(shape))
	{
		const TriangleMesh& mesh = std::get<TriangleMesh>(shape);
		vertex_buffer.AddArrayAndConvert<Vector3f>(mesh.positions);
		index_buffer.AddArray(mesh.indices);
		if (mesh.uvs.size() > 0) {
			uv_buffer.AddArrayAndConvert<Vector2f>(mesh.uvs);
		}
		else {
			uv_buffer.AddZeros<Vector2f>(mesh.positions.size());
		}
		if (mesh.normals.size() > 0) {
			normal_buffer.AddArrayAndConvert<Vector3f>(mesh.normals);
		}
		else {
			normal_buffer.AddZeros<Vector3f>(mesh.positions.size());
		}
	}
	else if (std::holds_alternative<Sphere>(shape))
	{
		const Sphere& sphere = std::get<Sphere>(shape);
		VkAabbPositionsKHR aabb = {
			.minX = (float)(sphere.position.x - sphere.radius),
			.minY = (float)(sphere.position.y - sphere.radius),
			.minZ = (float)(sphere.position.z - sphere.radius),
			.maxX = (float)(sphere.position.x + sphere.radius),
			.maxY = (float)(sphere.position.y + sphere.radius),
			.maxZ = (float)(sphere.position.z + sphere.radius),
		};
		aabb_buffer.Add(aabb);
	}
}

void GPUDevice::add_shape(uint32_t index, const GPU::Shape& gpu_shape, const Shape& shape, uint32_t sphere_index) {
	VkAccelerationStructureGeometryKHR geometry =
		std::visit(get_shape_geometry_op{ vertex_buffer, index_buffer, aabb_buffer, gpu_shape, sphere_index }, shape);

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.geometryCount = 1,
		.pGeometries = &geometry,
	};
	uint32_t primitive_count = std::holds_alternative<TriangleMesh>(shape)
		? std::get<TriangleMesh>(shape).indices.size() : 1;
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
		.instanceShaderBindingTableRecordOffset = std::holds_alternative<TriangleMesh>(shape) ? 0u : 1u,
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

GPU::TableDist2D GPUDevice::add_dist_2d(const TableDist2D& table)
{
	uint32_t pdf_rows_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.pdf_rows);

	uint32_t pdf_marg_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.pdf_marginals);

	uint32_t cdf_rows_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.cdf_rows);

	uint32_t cdf_marg_offset = dist_buffer.data.size() / sizeof(float);
	dist_buffer.AddArrayAndConvert<float>(table.cdf_marginals);

	GPU::TableDist2D result;
	result._pdf_cdf_rows = {
		pdf_rows_offset, (uint32_t)(table.pdf_rows.size()), cdf_rows_offset, (uint32_t)(table.cdf_rows.size())
	};
	result._pdf_cdf_marginals = {
		pdf_marg_offset, (uint32_t)(table.pdf_marginals.size()), cdf_marg_offset, (uint32_t)(table.cdf_marginals.size())
	};
	result.width = table.width;
	result.height = table.height;
	result.total_values = table.total_values;
	return result;
}

void GPUDevice::attach(RGFW_window* window, Scene* scene, const std::string &path)
{
	QueueFamilyIndices indices;
	VkPipelineShaderStageCreateInfo stages[6] = {};

	scene_name = fs::path(path).stem().generic_string();

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
		scene_block.add_binding("media",    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		scene_block.add_binding("tsamp",    VK_DESCRIPTOR_TYPE_SAMPLER);
		scene_block.add_binding("textures", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GPU::MAX_TEXTURE_COUNT);
		scene_block.add_binding("vsamp",    VK_DESCRIPTOR_TYPE_SAMPLER);
		scene_block.add_binding("volumes",  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GPU::MAX_TEXTURE_COUNT);

		scene_block.shader_stages = VK_SHADER_STAGE_RAYGEN_BIT_KHR
			                      | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
			                      | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
		scene_block.CreateLayout(device);

		VkPushConstantRange push_range = {
			.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
			.size = sizeof(GPU::PerFrameInfo),
		};
		VkDescriptorSetLayout layouts[] = {
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
	LOG("Loading Shaders...");
	{
		// Pick integration shader to use
		if (scene->options.integrator == Integrator::Path)
			stages[0] = load_shader_stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rgen_path_tracing");
		else if (scene->options.integrator == Integrator::VolPath)
			stages[0] = load_shader_stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR, "rgen_vol_path_tracing");

		stages[1] = load_shader_stage(VK_SHADER_STAGE_MISS_BIT_KHR, "miss");
		stages[2] = load_shader_stage(VK_SHADER_STAGE_MISS_BIT_KHR, "miss_shadow");
		stages[3] = load_shader_stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "chit_triangle");
		stages[4] = load_shader_stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, "chit_sphere");
		stages[5] = load_shader_stage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR, "ints_sphere");
	}
	LOG("Creating Ray Tracing Pipeline...");
	{
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
			{
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
				.generalShader      = VK_SHADER_UNUSED_KHR,
				.closestHitShader   = 4,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = 5,
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
	LOG("Create Samplers...");
	{
		VkSamplerCreateInfo sampler_info = {
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		};
		assert(vkCreateSampler(device, &sampler_info, 0, &sampler) == VK_SUCCESS);

		sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		assert(vkCreateSampler(device, &sampler_info, 0, &volume_sampler) == VK_SUCCESS);
	}
	LOG("Creating Scene Buffers...");
	{
		GPU::SceneInfo scene_info = {};
		{
			for (const Mipmap3& texture : scene->texture_pool.image3s) {
				add_texture(texture);
			}
			for (Medium &medium: (std::vector<Medium>&)scene->media) {
				if (!std::holds_alternative<HeterogeneousMedium>(medium))
					continue;

				auto& m = std::get<HeterogeneousMedium>(medium);
				using Volume = GridVolume<Spectrum>;

				if (std::holds_alternative<Volume>(m.albedo))
				{
					m.albedo_volume_id = add_volume(std::get<Volume>(m.albedo));
				}
				if (std::holds_alternative<Volume>(m.density))
				{
					m.density_volume_id = add_volume(std::get<Volume>(m.density));
				}
			}

			if (scene->envmap_light_id != -1) {
				scene_info.envmap = convert(*this, std::get<Envmap>(scene->lights[scene->envmap_light_id]));
			}
		}
		{
			for (const Material& material : scene->materials) {
				std::visit(material_convert_op{ material_buffer, texture_buffer }, material);
			}
			material_buffer.Create(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
			texture_buffer.Create(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
			aabb_buffer.CreateFromStaging(*this, usage);
			aabb_buffer.GetDeviceAddress(device);

			uv_buffer.CreateFromStaging(*this, usage);
			normal_buffer.CreateFromStaging(*this, usage);
		}
		{
			uint32_t vertex_offset = 0, index_offset = 0, sphere_index = 0;
			for (uint32_t i = 0; i < scene->shapes.size(); ++i) {
				const Shape& shape = scene->shapes[i];
				GPU::Shape gpu_shape = std::visit(shape_convert_op{ vertex_offset, index_offset }, shape);
				add_shape(i, gpu_shape, shape, sphere_index);
				if (std::holds_alternative<Sphere>(shape)) {
					sphere_index += 1;
				}
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
			for (const Medium &medium : scene->media) {
				media_buffer.Add(std::visit(medium_convert_op{}, medium));
			}
			media_buffer.CreateFromStaging(*this, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		}
		{
			scene_info.camera = convert(scene->camera);
			scene_info.bounds.center = scene->bounds.center;
			scene_info.bounds.radius = scene->bounds.radius;
			scene_info.options.max_depth = scene->options.max_depth;
			scene_info.options.rr_depth = scene->options.rr_depth;
			scene_info.options.max_null_collisions = scene->options.max_null_collisions;
			scene_info.envmap.light_id = scene->envmap_light_id;
			scene_info.options.vol_path_version = scene->options.vol_path_version;
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
		uint32_t primitive_count = (uint32_t)(bass.size());
		VkAccelerationStructureBuildSizesInfoKHR tSizes{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
		vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tBuildInfo, &primitive_count, &tSizes);

		createBuffer(tSizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tas.buffer.buffer, tas.buffer.memory);
		VkAccelerationStructureCreateInfoKHR tcreate{ VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR }; tcreate.buffer = tas.buffer.buffer; tcreate.size = tSizes.accelerationStructureSize; tcreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		vkCreateAccelerationStructureKHR(device, &tcreate, nullptr, &tas.handle);

		VkAccelerationStructureBuildRangeInfoKHR build_range = {
			.primitiveCount = primitive_count,
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

		uint32_t groupCount = 5;
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
		chit_sbt = { .deviceAddress = sbtAddr + 3 * baseAlignment, .stride = baseAlignment, .size = 2 * baseAlignment };
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
			{ VK_DESCRIPTOR_TYPE_SAMPLER,                     4 },
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

		std::vector<VkDescriptorImageInfo> texture_infos;
		for (VulkanImage& image : textures) {
			texture_infos.emplace_back(VkDescriptorImageInfo{
				.imageView = image.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			});
		}
		std::vector<VkDescriptorImageInfo> volume_infos;
		for (VulkanImage& image : volumes) {
			volume_infos.emplace_back(VkDescriptorImageInfo{
				.imageView = image.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			});
		}
		std::vector<VkWriteDescriptorSet> writes = {
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
			scene_block.write("tex",      texture_buffer),
			scene_block.write("dist",     dist_buffer),
			scene_block.write("media",    media_buffer),
			scene_block.write("tsamp",    sampler),
			scene_block.write("vsamp",    volume_sampler),
		};
		if (texture_infos.size() > 0)
			writes.emplace_back(scene_block.write_many("textures", texture_infos.data(), texture_infos.size()));
		if (volume_infos.size() > 0)
			writes.emplace_back(scene_block.write_many("volumes", volume_infos.data(), volume_infos.size()));
		vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, 0);
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

	for (const auto& stage : stages)
		if (stage.module) vkDestroyShaderModule(device, stage.module, 0);

	target_sample_count = (uint32_t)(scene->options.samples_per_pixel);
}

void GPUDevice::render(RGFW_window* window)
{
	Timer timer;
	tick(timer);

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
					ImGui::Text("%.3f M Rays Generated/second", (ImGui::GetIO().Framerate * float(width * height))/1e6f);
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
			Real render_time = tick(timer);
			int width = (int)(image_extent.width);
			int height = (int)(image_extent.height);
			char filename[256] = {};
			snprintf(filename, 256, "gpu_%s_%uspp.exr", scene_name.c_str(), frame_count + 1);
			Image3 image(width, height);
			struct Pixel {
				Vector3f rgb;
				float a;

				Vector3 convert() {
					if (isnan(rgb.x) || isnan(rgb.y) || isnan(rgb.z))
						return Vector3(0.0, 1.0, 1.0);
					return Vector3(rgb);
				}
			};
			Pixel* pixels = (Pixel*)storage_mapped;
			for (int y = 0; y < height; ++y)
				for (int x = 0; x < width; ++x)
					image(x, y) = pixels[y * width + x].convert();
			imwrite(filename, image);
			//imwrite_raw(filename, storage_mapped, width, height);
			LOG("Saved to \"%s\" (took %.3f seconds) (%.1f image samples/sec)", filename, render_time, (Real)(frame_count + 1)/render_time);
		}

		frame_count += 1;
	}

	vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

void VulkanRawBuffer::Create(GPUDevice& gpu, VkFlags usage)
{
	if (data.size() == 0) return;
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
	if (buffer) device_address = get_buffer_device_address(device, buffer);
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
	vkCmdDispatch(cmd, ceil_div(width, 8u), ceil_div(height, 8u), 1);
}

void GPUDevice::write_image(VkImage dstImage, const Image3& source, VkFormat format, VkImageLayout outLayout, int level)
{
	VkDeviceSize  data_size = source.width * source.height * format_size(format);
	VulkanBuffer staging{};

	// 2. Copy Image Data to Staging Buffer
	{
		createBuffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging.buffer, staging.memory);
		Vector4f* data;
		vkMapMemory(device, staging.memory, 0, data_size, 0, (void**)&data);
		int pixel_count = source.width * source.height;
		for (int i = 0; i < pixel_count; ++i) {
			Vector3f color = source(i);
			data[i] = Vector4f(color.x, color.y, color.z, 1.0f);
		}
		vkUnmapMemory(device, staging.memory);
	}

	// 3. Allocate and Begin writing to a Command Buffer for Transfer operations
	VkCommandBuffer cmd;
	VkCommandBufferAllocateInfo commandBufferInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferInfo.commandBufferCount = 1;
	commandBufferInfo.commandPool = command_pool;
	commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	vkAllocateCommandBuffers(device, &commandBufferInfo, &cmd);
	{
		VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &beginInfo);

		// 4. Set Image Layout for transfer
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = level;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
		VkImageMemoryBarrier imageBarrier_toTransfer = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = dstImage;
		imageBarrier_toTransfer.subresourceRange = range;
		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);

		// 5. Copy from Staging Buffer to Destination Image
		VkBufferImageCopy copyRegion{};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = level;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = { (uint32_t)(source.width), (uint32_t)(source.height), 1 };
		vkCmdCopyBufferToImage(cmd, staging.buffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		// 6. Set Image Layout to the Output Layout
		VkImageMemoryBarrier imageBarrier_toReadable{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imageBarrier_toReadable.image = dstImage;
		imageBarrier_toReadable.subresourceRange = range;
		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toReadable.newLayout = outLayout;
		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
		vkEndCommandBuffer(cmd);

		// 7. Sumbit commands to GPU and wait for them to complete
		VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmd;
		vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);
		staging.Destroy(device);
	}
	vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

void GPUDevice::write_volume(VkImage dstImage, const GridVolume<Spectrum>& source, VkFormat format, VkImageLayout outLayout)
{
	VkDeviceSize  data_size = source.resolution.x * source.resolution.y * source.resolution.z * format_size(format);
	VulkanBuffer staging{};

	// 2. Copy Image Data to Staging Buffer
	{
		createBuffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging.buffer, staging.memory);
		Vector4f* data;
		vkMapMemory(device, staging.memory, 0, data_size, 0, (void**)&data);
		int pixel_count = source.resolution.x * source.resolution.y * source.resolution.z;
		for (int i = 0; i < pixel_count; ++i) {
			Vector3f color = source.data[i];
			data[i] = Vector4f(color.x, color.y, color.z, 1.0f);
		}
		vkUnmapMemory(device, staging.memory);
	}

	// 3. Allocate and Begin writing to a Command Buffer for Transfer operations
	VkCommandBuffer cmd;
	VkCommandBufferAllocateInfo commandBufferInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	commandBufferInfo.commandBufferCount = 1;
	commandBufferInfo.commandPool = command_pool;
	commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	vkAllocateCommandBuffers(device, &commandBufferInfo, &cmd);
	{
		VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &beginInfo);

		// 4. Set Image Layout for transfer
		VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = 1;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
		VkImageMemoryBarrier imageBarrier_toTransfer = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imageBarrier_toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier_toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toTransfer.image = dstImage;
		imageBarrier_toTransfer.subresourceRange = range;
		imageBarrier_toTransfer.srcAccessMask = 0;
		imageBarrier_toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toTransfer);

		// 5. Copy from Staging Buffer to Destination Image
		VkBufferImageCopy copyRegion{};
		copyRegion.bufferOffset = 0;
		copyRegion.bufferRowLength = 0;
		copyRegion.bufferImageHeight = 0;
		copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copyRegion.imageSubresource.mipLevel = 0;
		copyRegion.imageSubresource.baseArrayLayer = 0;
		copyRegion.imageSubresource.layerCount = 1;
		copyRegion.imageExtent = {
			(uint32_t)(source.resolution.x),
			(uint32_t)(source.resolution.y),
			(uint32_t)(source.resolution.z)
		};
		vkCmdCopyBufferToImage(cmd, staging.buffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		// 6. Set Image Layout to the Output Layout
		VkImageMemoryBarrier imageBarrier_toReadable{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
		imageBarrier_toReadable.image = dstImage;
		imageBarrier_toReadable.subresourceRange = range;
		imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageBarrier_toReadable.newLayout = outLayout;
		imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
		vkEndCommandBuffer(cmd);

		// 7. Sumbit commands to GPU and wait for them to complete
		VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmd;
		vkQueueSubmit(graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);
		staging.Destroy(device);
	}
	vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

// ImGui does not recommend putting itself in a DLL,
// so I guess this will have to stay here...
#include "3rdparty/imgui.cpp"
#include "3rdparty/imgui_demo.cpp"
#include "3rdparty/imgui_draw.cpp"
#include "3rdparty/imgui_tables.cpp"
#include "3rdparty/imgui_widgets.cpp"
#include "3rdparty/imgui_impl_vulkan.cpp"
