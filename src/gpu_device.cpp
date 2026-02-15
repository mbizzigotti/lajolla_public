#include "gpu_device.h"
#include "shaders/basic.slang.inl"

/* References */
/*
	https://docs.vulkan.org/spec/latest/chapters/pipelines.html#_ray_tracing_pipeline_creation
	https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/extensions/ray_tracing_basic/ray_tracing_basic.cpp
*/

namespace vk {
	PFN_vkCreateRayTracingPipelinesKHR CreateRayTracingPipelinesKHR;
	PFN_vkCmdTraceRaysKHR              CmdTraceRaysKHR;
}

VkPipelineShaderStageCreateInfo load_shader(VkShaderModule shader, VkShaderStageFlagBits stage) {
	VkPipelineShaderStageCreateInfo shader_stage = {};
	shader_stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shader_stage.stage  = stage;
	shader_stage.module = shader;
	shader_stage.pName  = "main";
	return shader_stage;
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
			"VK_KHR_dynamic_rendering",
			"VK_KHR_deferred_host_operations",
			"VK_KHR_acceleration_structure",
			"VK_KHR_ray_tracing_pipeline",
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

		VkDeviceCreateInfo device_info = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext = &ray_tracing_feature,
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &queue_info,
			.enabledLayerCount = enable_validation ? std::size(layer_names) : 0,
			.ppEnabledLayerNames = layer_names,
			.enabledExtensionCount = std::size(device_extensions),
			.ppEnabledExtensionNames = device_extensions,
		};
		
		assert(vkCreateDevice(physical_device, &device_info, 0, &device) == VK_SUCCESS);

		vkGetDeviceQueue(device, 0, 0, &queue);

		assert(vk::CreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
		assert(vk::CmdTraceRaysKHR              = (PFN_vkCmdTraceRaysKHR)             vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
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
		// Slot for binding top level acceleration structures to the ray generation shader
		VkDescriptorSetLayoutBinding acceleration_structure_layout_binding{};
		acceleration_structure_layout_binding.binding         = 0;
		acceleration_structure_layout_binding.descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		acceleration_structure_layout_binding.descriptorCount = 1;
		acceleration_structure_layout_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

		VkDescriptorSetLayoutBinding result_image_layout_binding{};
		result_image_layout_binding.binding         = 1;
		result_image_layout_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		result_image_layout_binding.descriptorCount = 1;
		result_image_layout_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

		VkDescriptorSetLayoutBinding uniform_buffer_binding{};
		uniform_buffer_binding.binding         = 2;
		uniform_buffer_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniform_buffer_binding.descriptorCount = 1;
		uniform_buffer_binding.stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

		std::vector<VkDescriptorSetLayoutBinding> bindings = {
			/*
			acceleration_structure_layout_binding,
			result_image_layout_binding,
			uniform_buffer_binding
			*/
		};

		VkDescriptorSetLayoutCreateInfo layout_info{};
		layout_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
		layout_info.pBindings    = bindings.data();
		assert(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) == VK_SUCCESS);

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
		std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;

		// Ray generation group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR raygen_group_ci{};
			raygen_group_ci.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			raygen_group_ci.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			raygen_group_ci.generalShader      = static_cast<uint32_t>(shader_stages.size()) - 1;
			raygen_group_ci.closestHitShader   = VK_SHADER_UNUSED_KHR;
			raygen_group_ci.anyHitShader       = VK_SHADER_UNUSED_KHR;
			raygen_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
			shader_groups.push_back(raygen_group_ci);
		}

		// Ray miss group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_MISS_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR miss_group_ci{};
			miss_group_ci.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			miss_group_ci.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			miss_group_ci.generalShader      = static_cast<uint32_t>(shader_stages.size()) - 1;
			miss_group_ci.closestHitShader   = VK_SHADER_UNUSED_KHR;
			miss_group_ci.anyHitShader       = VK_SHADER_UNUSED_KHR;
			miss_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
			shader_groups.push_back(miss_group_ci);
		}

		// Ray closest hit group
		{
			shader_stages.push_back(load_shader(shader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
			VkRayTracingShaderGroupCreateInfoKHR closes_hit_group_ci{};
			closes_hit_group_ci.sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
			closes_hit_group_ci.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
			closes_hit_group_ci.generalShader      = VK_SHADER_UNUSED_KHR;
			closes_hit_group_ci.closestHitShader   = static_cast<uint32_t>(shader_stages.size()) - 1;
			closes_hit_group_ci.anyHitShader       = VK_SHADER_UNUSED_KHR;
			closes_hit_group_ci.intersectionShader = VK_SHADER_UNUSED_KHR;
			shader_groups.push_back(closes_hit_group_ci);
		}

		/*
			Create the ray tracing pipeline
		*/
		VkRayTracingPipelineCreateInfoKHR raytracing_pipeline_create_info{};
		raytracing_pipeline_create_info.sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
		raytracing_pipeline_create_info.stageCount                   = static_cast<uint32_t>(shader_stages.size());
		raytracing_pipeline_create_info.pStages                      = shader_stages.data();
		raytracing_pipeline_create_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
		raytracing_pipeline_create_info.pGroups                      = shader_groups.data();
		raytracing_pipeline_create_info.maxPipelineRayRecursionDepth = 1;
		raytracing_pipeline_create_info.layout                       = pipeline_layout;
		assert(vk::CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &raytracing_pipeline_create_info, nullptr, &pipeline) == VK_SUCCESS);
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
					.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

			#if 0
			VkClearValue clear_value {};
			clear_value.color.float32[0] = 1.0f;
			clear_value.color.float32[1] = 0.0f;
			clear_value.color.float32[2] = 0.0f;
			clear_value.color.float32[3] = 1.0f;

			const VkRenderingAttachmentInfo color_attachment_info {
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = image_views[image_index],
				.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = clear_value,
			};
			
			const VkRenderingInfo render_info {
				.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
				.renderArea = {
					.extent = image_extent,
				},
				.layerCount = 1,
				.colorAttachmentCount = 1,
				.pColorAttachments = &color_attachment_info,
			};
			
			vkCmdBeginRendering(command_buffer, &render_info);
			#endif

			VkStridedDeviceAddressRegionKHR raygen_shader_binding_table {};

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
			//vk::CmdTraceRaysKHR(
			//	command_buffer,
			//	&raygen_shader_binding_table,
			//	&raygen_shader_binding_table,
			//	&raygen_shader_binding_table,
			//	&raygen_shader_binding_table,
			//	100, 100, 1);

			#if 0
			vkCmdEndRendering(command_buffer);
			#endif

			{
				const VkImageMemoryBarrier image_memory_barrier {
					.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
					.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
