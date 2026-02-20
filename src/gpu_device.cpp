#include "gpu_device.h"
#include "flexception.h"
#include "shaders/basic.slang.inl"

/* References */
/*
	https://docs.vulkan.org/spec/latest/chapters/pipelines.html#_ray_tracing_pipeline_creation
	https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/ray_tracing_basic/ray_tracing_basic.cpp
*/

VkPipelineShaderStageCreateInfo load_shader(VkShaderModule shader, VkShaderStageFlagBits stage) {
	VkPipelineShaderStageCreateInfo shader_stage = {};
	shader_stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage.stage  = stage;
	shader_stage.module = shader;
	shader_stage.pName  = "main";
	return shader_stage;
}

inline uint32_t align(uint32_t value, uint32_t alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}

void GPUDevice::create_buffer(VkBuffer* buffer, VkDeviceMemory* memory, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags)
{
	VkBufferCreateInfo buffer_info {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
	};
	assert(vkCreateBuffer(device, &buffer_info, nullptr, buffer) == VK_SUCCESS);

	VkMemoryRequirements mem_requirements;
	vkGetBufferMemoryRequirements(device, *buffer, &mem_requirements);

	VkMemoryAllocateInfo alloc_info {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = mem_requirements.size,
		.memoryTypeIndex = find_memory_type(mem_requirements.memoryTypeBits, memory_flags),
	};
	assert(vkAllocateMemory(device, &alloc_info, nullptr, memory) == VK_SUCCESS);
	assert(vkBindBufferMemory(device, *buffer, *memory, 0) == VK_SUCCESS);
}

uint32_t GPUDevice::find_memory_type(uint32_t type_filter, uint32_t desired_flags)
{
	VkPhysicalDeviceMemoryProperties mem_properties;
	vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
	for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
	{
		if ((type_filter & (1 << i)) &&
			(mem_properties.memoryTypes[i].propertyFlags & desired_flags) == desired_flags)
		{
			return i;
		}
	}
	return VK_MAX_MEMORY_TYPES;
}

VkDeviceAddress GPUDevice::get_device_address(VkBuffer buffer)
{
	VkBufferDeviceAddressInfoKHR buffer_device_address_info{
		.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer,
	};
	return vkGetBufferDeviceAddressKHR(device, &buffer_device_address_info);
}

GPUDevice::GPUDevice() {
	static constexpr bool enable_validation = true;

	std::cout << "Creating Vulkan instance..." << std::endl;
	{
		VkApplicationInfo application_info = {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.apiVersion = VK_API_VERSION_1_3,
		};

		const char* extension_names[] = {
		#if defined(__APPLE__)
			"VK_KHR_portability_enumeration",
		#endif
			VK_KHR_SURFACE_EXTENSION_NAME,
			RGFW_VK_SURFACE,
		};

		const char* layer_names[] = {
			"VK_LAYER_KHRONOS_validation",
		};
			
		VkInstanceCreateInfo instance_info = {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		#if defined(__APPLE__)
			.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
		#endif
			.pApplicationInfo = &application_info,
			.enabledLayerCount = enable_validation ? 1 : 0,
			.ppEnabledLayerNames = layer_names,
			.enabledExtensionCount = std::size(extension_names),
			.ppEnabledExtensionNames =  extension_names,
		};

		assert(vkCreateInstance(&instance_info, 0, &instance) == VK_SUCCESS);
	}
	
	std::cout << "Picking Vulkan physical device..." << std::endl;
	{
		uint32_t count = 1;
		assert(vkEnumeratePhysicalDevices(instance, &count, &physical_device) == VK_SUCCESS);
	}
	
	std::cout << "Creating Vulkan logical device..." << std::endl;
	{
		const char* device_extensions[] = {
		#if defined(__APPLE__)
			"VK_KHR_portability_subset",
		#endif
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
			VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
			VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
			VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
			VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
			VK_KHR_SPIRV_1_4_EXTENSION_NAME,
			VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
		};

		const char* layer_names[] = {
			"VK_LAYER_KHRONOS_validation",
		};

		float priority = 1.0f;
		VkDeviceQueueCreateInfo queue_info = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueCount = 1,
			.pQueuePriorities = &priority,
		};

		VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_rendering_feature {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
			.pNext = NULL,
			.dynamicRendering = VK_TRUE,
		};
		VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_feature {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
			.pNext = &dynamic_rendering_feature,
			.rayTracingPipeline = VK_TRUE,
		};
		VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_feature {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			.pNext = &ray_tracing_feature,
			.accelerationStructure = VK_TRUE,
		};
		VkPhysicalDeviceBufferDeviceAddressFeaturesEXT device_address_features {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT,
			.pNext = &acceleration_structure_feature,
			.bufferDeviceAddress = VK_TRUE,
		};

		VkDeviceCreateInfo device_info = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &device_address_features,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queue_info,
			.enabledLayerCount = enable_validation ? std::size(layer_names) : 0,
			.ppEnabledLayerNames = layer_names,
			.enabledExtensionCount = std::size(device_extensions),
			.ppEnabledExtensionNames = device_extensions,
		};
		
		VkResult result = vkCreateDevice(physical_device, &device_info, 0, &device);
		if (result != VK_SUCCESS) {
			if (result == VK_ERROR_EXTENSION_NOT_PRESENT)
				Error("Device extensions not present!");
			Error("Failed to create Vulkan device!");
		}

		vkGetDeviceQueue(device, 0, 0, &queue);

		#define LOAD(NAME) assert(NAME = (PFN_##NAME)vkGetDeviceProcAddr(device, #NAME))
		LOAD(vkCreateRayTracingPipelinesKHR);
		LOAD(vkCmdTraceRaysKHR);
		LOAD(vkGetBufferDeviceAddressKHR);
		LOAD(vkGetRayTracingShaderGroupHandlesKHR);
		#undef LOAD
	}

	
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
	};
	{
		VkPhysicalDeviceProperties2 device_properties {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext = &ray_tracing_pipeline_properties,
		};
		vkGetPhysicalDeviceProperties2(physical_device, &device_properties);

		VkPhysicalDeviceFeatures2 device_features {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &acceleration_structure_features,
		};
		vkGetPhysicalDeviceFeatures2(physical_device, &device_features);
	}
	
	std::cout << "Creating command buffers..." << std::endl;
	{
		VkCommandPoolCreateInfo graphics_pool_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = 0,
		};
		assert(vkCreateCommandPool(device, &graphics_pool_info, 0, &command_pool) == VK_SUCCESS);

		VkCommandBufferAllocateInfo allocate_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = command_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = (u32)std::size(command_buffers),
		};
		assert(vkAllocateCommandBuffers(device, &allocate_info, command_buffers) == VK_SUCCESS);
	}

	std::cout << "Creating syncronization objects..." << std::endl;
	{
		VkSemaphoreCreateInfo semaphore_info = {
			.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
		};
		
		for (uint32_t i = 0; i < (u32)std::size(present_ready_semaphores); ++i)
			vkCreateSemaphore(device, &semaphore_info, 0, &present_ready_semaphores[i]);

		VkFenceCreateInfo fence_info = {
			.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
			.flags = VK_FENCE_CREATE_SIGNALED_BIT,
		};
		
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
		{
			vkCreateSemaphore(device, &semaphore_info, 0, &image_ready_semaphores[i]);
			vkCreateFence(device, &fence_info, 0, &fences[i]);
		}
	}

	std::cout << "Creating ray tracing pipeline..." << std::endl;
	{
		VkDescriptorSetLayoutBinding bindings[] = {
			{ .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
			{ .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR },
		};
		
		for (uint32_t i = 0; i < std::size(bindings); ++i)
		{
			bindings[i].binding = i;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
		}

		VkDescriptorSetLayoutCreateInfo layout_info{};
		layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
		layout_info.pBindings    = bindings;
		assert(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) == VK_SUCCESS);

		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,              16 },
			{ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,             16 },
		};
		VkDescriptorPoolCreateInfo descriptor_pool_info = {
    		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    		.maxSets = 1,
    		.poolSizeCount = static_cast<uint32_t>(std::size(pool_sizes)),
    		.pPoolSizes = pool_sizes,
		};
		assert(vkCreateDescriptorPool(device, &descriptor_pool_info, 0, &descriptor_pool) == VK_SUCCESS);

		VkDescriptorSetAllocateInfo allocate_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    		.descriptorPool = descriptor_pool,
    		.descriptorSetCount = 1,
    		.pSetLayouts = &descriptor_set_layout,
		};
		assert(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set) == VK_SUCCESS);

		VkPipelineLayoutCreateInfo pipeline_layout_create_info{};
		pipeline_layout_create_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts    = &descriptor_set_layout;

		assert(vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) == VK_SUCCESS);

		VkShaderModule shader = 0;
		VkShaderModuleCreateInfo shader_info {
    		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    		.codeSize = basic_spv_len,
    		.pCode = reinterpret_cast<uint32_t*>(basic_spv),
		};
		assert(vkCreateShaderModule(device, &shader_info, 0, &shader) == VK_SUCCESS);

		/*
			Setup ray tracing shader groups
			Each shader group points at the corresponding shader in the pipeline
		*/
		std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
		VkRayTracingShaderGroupCreateInfoKHR shader_groups[3];

		// Ray generation group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
			shader_groups[SHADER_RAYGEN] = {
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
				.generalShader      = 0,
				.closestHitShader   = VK_SHADER_UNUSED_KHR,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			};
		}

		// Ray miss group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_MISS_BIT_KHR));
			shader_groups[SHADER_MISS] = {
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
				.generalShader      = 1,
				.closestHitShader   = VK_SHADER_UNUSED_KHR,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			};
		}

		// Ray closest hit group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
			shader_groups[SHADER_HIT] = {
				.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
				.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
				.generalShader      = VK_SHADER_UNUSED_KHR,
				.closestHitShader   = 2,
				.anyHitShader       = VK_SHADER_UNUSED_KHR,
				.intersectionShader = VK_SHADER_UNUSED_KHR,
			};
		}

		/*
			Create the ray tracing pipeline
		*/
		VkRayTracingPipelineCreateInfoKHR raytracing_pipeline_create_info{};
		raytracing_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
		raytracing_pipeline_create_info.stageCount                   = static_cast<uint32_t>(shader_stages.size());
		raytracing_pipeline_create_info.pStages                      = shader_stages.data();
		raytracing_pipeline_create_info.groupCount                   = static_cast<uint32_t>(std::size(shader_groups));
		raytracing_pipeline_create_info.pGroups                      = shader_groups;
		raytracing_pipeline_create_info.maxPipelineRayRecursionDepth = 1;
		raytracing_pipeline_create_info.layout                       = pipeline_layout;
		assert(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &raytracing_pipeline_create_info, nullptr, &pipeline) == VK_SUCCESS);
	}

	std::cout << "Creating shader binding tables..." << std::endl;
	{
		auto group_count = 3; // raygen + miss + hit
		auto handle_size = ray_tracing_pipeline_properties.shaderGroupHandleSize;
		auto aligned_handle_size = align(handle_size, ray_tracing_pipeline_properties.shaderGroupBaseAlignment);
		auto sbt_size = group_count * aligned_handle_size;
		
		auto usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
		auto properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		create_buffer(&sbt_buffer, &sbt_memory, sbt_size, usage, properties);

		// Copy the pipeline's shader handles into a host buffer
		void* sbt_data = 0;
		vkMapMemory(device, sbt_memory, 0, VK_WHOLE_SIZE, 0, &sbt_data);
		vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, group_count, sbt_size, sbt_data);
		
		uint32_t size = align(ray_tracing_pipeline_properties.shaderGroupHandleSize,
			                  ray_tracing_pipeline_properties.shaderGroupHandleAlignment);
		uint32_t device_address = get_device_address(sbt_buffer);
		
		shader_binding_tables[SHADER_RAYGEN] = { device_address + 0 * aligned_handle_size, size, size };
		shader_binding_tables[SHADER_MISS]   = { device_address + 1 * aligned_handle_size, size, size };
		shader_binding_tables[SHADER_HIT]    = { device_address + 2 * aligned_handle_size, size, size };
		shader_binding_tables[SHADER_CALLABLE] = {};
	}
}

void GPUDevice::attach(RGFW_window *window) {
	std::cout << "Creating Vulkan surface..." << std::endl;
	{
		assert(RGFW_window_createSurface_Vulkan(window, instance, &surface) == VK_SUCCESS);
	}
	
	std::cout << "Creating Vulkan swap chain..." << std::endl;
	{
		VkSurfaceCapabilitiesKHR capabilities = {0};
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);
		image_extent = capabilities.currentExtent;

		VkSurfaceFormatKHR surface_format;
		uint32_t count = 1;
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &count, &surface_format);
		format = surface_format.format;
		
		VkSwapchainCreateInfoKHR swap_chain_info = {
			.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
			.surface = surface,
			.minImageCount = capabilities.minImageCount + 1,
			.imageFormat = surface_format.format,
			.imageColorSpace = surface_format.colorSpace,
			.imageExtent = capabilities.currentExtent,
			.imageArrayLayers = 1, /* For non-stereoscopic-3D applications (non-VR), this value is 1 */
			.imageUsage = capabilities.supportedUsageFlags,
			.preTransform = capabilities.currentTransform,
			.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
			.presentMode = VK_PRESENT_MODE_FIFO_KHR,
			.clipped = VK_TRUE, /* "... allows more efficient presentation methods to be used on some platforms." */
		};

		assert(vkCreateSwapchainKHR(device, &swap_chain_info, 0, &swap_chain) == VK_SUCCESS);
	}
	
	std::cout << "Creating swap chain image views..." << std::endl;
	{
		vkGetSwapchainImagesKHR(device, swap_chain, &image_count, 0);
		assert(image_count > 0);
		assert(image_count <= MAX_SWAP_CHAIN_IMAGES);

		vkGetSwapchainImagesKHR(device, swap_chain, &image_count, images);

		for (uint32_t i = 0; i < image_count; ++i)
		{
			VkImageViewCreateInfo view_info = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = images[i],
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = format,
				.components = {}, // Identity
				.subresourceRange = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				}
			};
			assert(vkCreateImageView(device, &view_info, 0, image_views + i) == VK_SUCCESS);
		}
	}
}

void GPUDevice::render(RGFW_window *window, Scene& scene) {
	build_acceleration_structure();

	bool running = true;
	uint32_t frame_count = 0;
    while (!RGFW_window_shouldClose(window) && running)
	{
		RGFW_event event;
		while (RGFW_window_checkEvent(window, &event)) {
			if (event.type == RGFW_quit) {
				running = false;
				break;
			}
		}
		
		uint32_t frame_index = frame_count % MAX_FRAMES_IN_FLIGHT;
		VkFence fence = fences[frame_index];
		VkSemaphore image_ready_semaphore = image_ready_semaphores[frame_index];
	
    	vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
    	vkResetFences(device, 1, &fence);

		uint32_t image_index = 0;
		VkResult result = VK_ERROR_UNKNOWN;
		while (result != VK_SUCCESS) {
			result = vkAcquireNextImageKHR(device, swap_chain,
				UINT64_MAX, image_ready_semaphore, VK_NULL_HANDLE, &image_index);

			if (result == VK_SUBOPTIMAL_KHR) break;
			else if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			//	resize();
				continue;
			}
			else if (result != VK_SUCCESS) {
				printf("Failed to get swap chain image! (%i)\n", result);
				exit(1);
			}
		}
	
		VkCommandBuffer command_buffer = command_buffers[frame_index];
		VkCommandBufferBeginInfo cmd_begin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		};
		vkBeginCommandBuffer(command_buffer, &cmd_begin);
		{
			{
				const VkImageMemoryBarrier image_memory_barrier {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
					.newLayout = VK_IMAGE_LAYOUT_GENERAL,
					.image = images[image_index],
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					}
				};

				vkCmdPipelineBarrier(
					command_buffer,
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,  // srcStageMask
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // dstStageMask
					0,
					0,
					nullptr,
					0,
					nullptr,
					1, // imageMemoryBarrierCount
					&image_memory_barrier // pImageMemoryBarriers
				);
			}

			VkDescriptorImageInfo image_info = {
				.imageView = image_views[image_index],
				.imageLayout = VK_IMAGE_LAYOUT_GENERAL,
			};
			VkWriteDescriptorSet write = {
    			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    			.dstSet = descriptor_set,
    			.dstBinding = 0,
    			.descriptorCount = 1,
    			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    			.pImageInfo = &image_info,
			};
			vkUpdateDescriptorSets(device, 1, &write, 0, 0);

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
				pipeline_layout, 0, 1, &descriptor_set, 0, 0);
			vkCmdTraceRaysKHR(command_buffer,
				&shader_binding_tables[SHADER_RAYGEN],
				&shader_binding_tables[SHADER_MISS],
				&shader_binding_tables[SHADER_HIT],
				&shader_binding_tables[SHADER_CALLABLE],
				image_extent.width, image_extent.height, 1);

			{
				const VkImageMemoryBarrier image_memory_barrier {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
					.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					.image = images[image_index],
					.subresourceRange = {
						.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1,
					}
				};
				vkCmdPipelineBarrier(
					command_buffer,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  // srcStageMask
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, // dstStageMask
					0,
					0,
					nullptr,
					0,
					nullptr,
					1, // imageMemoryBarrierCount
					&image_memory_barrier // pImageMemoryBarriers
				);
			}
		}
		vkEndCommandBuffer(command_buffer);

		VkSemaphore present_ready_semaphore = present_ready_semaphores[image_index];
		VkPipelineStageFlags wait_mask = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSubmitInfo submit_info = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &image_ready_semaphore,
			.pWaitDstStageMask = &wait_mask,
			.commandBufferCount = 1,
			.pCommandBuffers = &command_buffer,
			.signalSemaphoreCount = 1,
			.pSignalSemaphores = &present_ready_semaphore,
		};
		vkQueueSubmit(queue, 1, &submit_info, fence);

		VkPresentInfoKHR present_info = {
			.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &present_ready_semaphore,
			.swapchainCount = 1,
			.pSwapchains = &swap_chain,
			.pImageIndices = &image_index,
		};
		vkQueuePresentKHR(queue, &present_info);
	}
}

void GPUDevice::build_acceleration_structure()
{
	VkWriteDescriptorSetAccelerationStructureKHR as_info {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
		.accelerationStructureCount = 1,
		.pAccelerationStructures = &top_level_acceleration_structure.handle,
	};
	VkWriteDescriptorSet write {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = &as_info,
		.dstSet = descriptor_set,
		.dstBinding = 1,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
	};
	vkUpdateDescriptorSets(device, 1, &write, 0, 0);
}
