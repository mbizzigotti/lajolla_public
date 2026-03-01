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

struct VulkanBuffer {
	VkBuffer       buffer{ 0 };
	VkDeviceMemory memory{ 0 };

	operator VkBuffer() { return buffer; }

	void Create(struct GPUDevice &gpu, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags);
	void Destroy(VkDevice device);
};

#define MAX_SWAP_CHAIN_IMAGES 4 /* For mobile devices, this would need to be higher.. */
struct GPUDevice {
	PFN_vkCreateRayTracingPipelinesKHR              vkCreateRayTracingPipelinesKHR             { nullptr };
	PFN_vkCmdTraceRaysKHR                           vkCmdTraceRaysKHR                          { nullptr };
	PFN_vkGetBufferDeviceAddressKHR                 vkGetBufferDeviceAddressKHR                { nullptr };
	PFN_vkGetRayTracingShaderGroupHandlesKHR        vkGetRayTracingShaderGroupHandlesKHR       { nullptr };
	PFN_vkCreateAccelerationStructureKHR            vkCreateAccelerationStructureKHR           { nullptr };
	PFN_vkDestroyAccelerationStructureKHR           vkDestroyAccelerationStructureKHR          { nullptr };
	PFN_vkCmdBuildAccelerationStructuresKHR         vkCmdBuildAccelerationStructuresKHR        { nullptr };
	PFN_vkGetAccelerationStructureDeviceAddressKHR  vkGetAccelerationStructureDeviceAddressKHR { nullptr };
	PFN_vkGetAccelerationStructureBuildSizesKHR     vkGetAccelerationStructureBuildSizesKHR    { nullptr };

	VkInstance         instance{ 0 };
	VkSurfaceKHR       surface{ 0 };
	VkPhysicalDevice   physical_device{ 0 };
	VkDevice           device{ 0 };
	VkQueue            queue{ 0 };
	VkSwapchainKHR     swap_chain{ 0 };
	VkFormat           format{ VK_FORMAT_UNDEFINED };
	VkExtent2D         image_extent{ 0 };
	VkCommandPool      command_pool{ 0 };
	VkCommandBuffer    command_buffer{ 0 };
	uint32_t           image_count{ 0 };
	VkImage            images[MAX_SWAP_CHAIN_IMAGES]{ 0 };
	VkImageView        image_views[MAX_SWAP_CHAIN_IMAGES]{ 0 };
	VkSemaphore        render_finished_semaphore{ 0 };
	VkSemaphore        image_available_semaphore{ 0 };
	VkFence            fence{ 0 };
	VkDeviceMemory     storage_memory{ 0 };
	VkImage            storage_image{ 0 };
	VkImageView        storage_view{ 0 };

	struct AccelerationStructure
	{
		VkAccelerationStructureKHR handle;
		VkDeviceAddress            device_address;
		VulkanBuffer               buffer;
	};

	VkDescriptorPool      descriptor_pool{ 0 };
	VkDescriptorSet       descriptor_set{ 0 };
	VkDescriptorSetLayout descriptor_set_layout{ 0 };
	VkPipelineLayout      pipeline_layout{ 0 };
	VkPipeline            pipeline{ 0 };

	AccelerationStructure           bas{}, tas{};
	VulkanBuffer                    sbt_raygen{ 0 };
	VulkanBuffer                    sbt_miss{ 0 };
	VulkanBuffer                    sbt_hit{ 0 };
	VkStridedDeviceAddressRegionKHR shader_binding_tables[SHADER_COUNT]{ 0 };

	struct UniformData
	{
		Matrix4x4f view_inverse;
		Matrix4x4f proj_inverse;
	} uniform_data{};

	VulkanBuffer vertex_buffer{ 0 };
	VulkanBuffer index_buffer{ 0 };
	VkBuffer ubo{ 0 };

	GPUDevice();
	~GPUDevice();

	void attach(RGFW_window* window, Scene *scene);
	void render(RGFW_window *window);

private:

};
