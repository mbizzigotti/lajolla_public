#pragma once
#include "scene.h"
#define RGFW_IMPORT
#define RGFW_VULKAN
#define NOMINMAX
#include "3rdparty/RGFW.h"

enum {
	SHADER_RAYGEN,
	SHADER_MISS,
	SHADER_HIT,
	SHADER_CALLABLE,
	SHADER_COUNT,
};

#define MAX_FRAMES_IN_FLIGHT  2
#define MAX_SWAP_CHAIN_IMAGES 4 /* For mobile devices, this would need to be higher.. */
struct GPUDevice {
	PFN_vkCreateRayTracingPipelinesKHR       vkCreateRayTracingPipelinesKHR;
	PFN_vkCmdTraceRaysKHR                    vkCmdTraceRaysKHR;
	PFN_vkGetBufferDeviceAddressKHR          vkGetBufferDeviceAddressKHR;
	PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;

	VkInstance         instance;
	VkSurfaceKHR       surface;
	VkPhysicalDevice   physical_device;
	VkDevice           device;
	VkQueue            queue;
	VkSwapchainKHR     swap_chain;
	VkFormat           format;
	VkExtent2D         image_extent;
	VkCommandPool      command_pool;
	VkCommandBuffer    command_buffers[MAX_FRAMES_IN_FLIGHT];
	uint32_t           image_count;
	VkImage            images[MAX_SWAP_CHAIN_IMAGES];
	VkImageView        image_views[MAX_SWAP_CHAIN_IMAGES];
	VkSemaphore        image_ready_semaphores[MAX_FRAMES_IN_FLIGHT];    // Signaled when image is ready to be rendered to
	VkSemaphore        present_ready_semaphores[MAX_SWAP_CHAIN_IMAGES]; // Signaled when image is ready to be presented
	VkFence            fences[MAX_FRAMES_IN_FLIGHT];

	struct AccelerationStructure
	{
		VkAccelerationStructureKHR handle;
		VkDeviceAddress            device_address;
		VkBuffer                   buffer;
	};

	VkDescriptorPool      descriptor_pool;
	VkDescriptorSet       descriptor_set;
	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout      pipeline_layout;
	VkPipeline            pipeline;

	AccelerationStructure           bottom_level_acceleration_structure{};
	AccelerationStructure           top_level_acceleration_structure{};
	VkBuffer                        sbt_buffer;
	VkDeviceMemory                  sbt_memory;
	VkStridedDeviceAddressRegionKHR shader_binding_tables[SHADER_COUNT];

	struct UniformData
	{
		Matrix4x4f view_inverse;
		Matrix4x4f proj_inverse;
	} uniform_data;

	VkBuffer ubo;

	GPUDevice();

	void attach(RGFW_window *window);
	void render(RGFW_window *window, Scene& scene);
	void build_acceleration_structure();

private:
	void create_buffer(VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags);
	uint32_t find_memory_type(uint32_t type_filter, uint32_t desired_flags);
	VkDeviceAddress get_device_address(VkBuffer buffer);
};
