#pragma once
#include "scene.h"
#define RGFW_IMPORT
#define RGFW_VULKAN
#define NOMINMAX
#include "3rdparty/RGFW.h"

struct VulkanBuffer {
	VkBuffer       buffer{ 0 };
	VkDeviceMemory memory{ 0 };

	operator VkBuffer() { return buffer; }

	void Create(struct GPUDevice &gpu, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags);
	void Destroy(VkDevice device);
};

struct VulkanAccelerationStructure {
	VkAccelerationStructureKHR handle{ 0 };
	VkDeviceAddress            address{ 0 };
	VulkanBuffer               buffer{ 0 };

	void Create(struct GPUDevice& gpu);
	void Destroy(struct GPUDevice& gpu);
};

struct VulkanTriangleMesh {
	VulkanBuffer vertex_buffer{ 0 };
	VulkanBuffer index_buffer{ 0 };
};

struct VulkanRawBuffer {
	std::vector<uint8_t> data;
	VulkanBuffer buffer;

	void Create(struct GPUDevice& gpu, VkFlags usage);
	void Destroy(struct GPUDevice& gpu);

	template <typename T>
	void Add(const T& t) {
		uint8_t* bytes = (uint8_t*)(&t);
		uint32_t size = sizeof(T);
		for (uint32_t i = 0; i < size; ++i)
			data.emplace_back(bytes[i]);
	}
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
	PFN_vkBuildAccelerationStructuresKHR            vkBuildAccelerationStructuresKHR           { nullptr };
	PFN_vkGetAccelerationStructureDeviceAddressKHR  vkGetAccelerationStructureDeviceAddressKHR { nullptr };
	PFN_vkGetAccelerationStructureBuildSizesKHR     vkGetAccelerationStructureBuildSizesKHR    { nullptr };

	VkInstance                      instance{ 0 };
	VkSurfaceKHR                    surface{ 0 };
	VkPhysicalDevice                physical_device{ 0 };
	VkDevice                        device{ 0 };
	VkQueue                         graphics_queue{ 0 };
	VkQueue                         present_queue{ 0 };
	VkSwapchainKHR                  swap_chain{ 0 };
	VkFormat                        swap_format{ VK_FORMAT_UNDEFINED };
	VkExtent2D                      image_extent{ 0 };
	VkCommandPool                   command_pool{ 0 };
	VkCommandBuffer                 command_buffer{ 0 };
	uint32_t                        image_count{ 0 };
	VkImage                         images[MAX_SWAP_CHAIN_IMAGES]{ 0 };
	VkImageView                     image_views[MAX_SWAP_CHAIN_IMAGES]{ 0 };
	VkSemaphore                     render_finished_semaphore{ 0 };
	VkSemaphore                     image_available_semaphore{ 0 };
	VkFence                         fence{ 0 };
	VkDeviceMemory                  storage_memory{ 0 };
	VkImage                         storage_image{ 0 };
	VkImageView                     storage_view{ 0 };
	VkDescriptorPool                descriptor_pool{ 0 };
	VkDescriptorSet                 descriptor_set{ 0 };
	VkDescriptorSetLayout           descriptor_set_layout{ 0 };
	VkPipelineLayout                pipeline_layout{ 0 };
	VkPipeline                      pipeline{ 0 };
	VulkanAccelerationStructure     tas{};
	VulkanBuffer                    sbt_buffer{ 0 };
	VkStridedDeviceAddressRegionKHR rgen_sbt{};
	VkStridedDeviceAddressRegionKHR miss_sbt{};
	VkStridedDeviceAddressRegionKHR chit_sbt{};
	VulkanBuffer                    instance_buffer{};
	VulkanBuffer                    camera_buffer{};
	VulkanRawBuffer                 material_buffer{};
	
	std::vector<VkAccelerationStructureInstanceKHR>        instances;
	std::vector<VulkanAccelerationStructure>               bass;
	std::vector<VulkanTriangleMesh>                        triangle_meshes;
	
	struct UniformData
	{
		Matrix4x4f view_inverse;
		Matrix4x4f proj_inverse;
	} uniform_data{};

	VkBuffer ubo{ 0 };

	GPUDevice();
	~GPUDevice();

	void attach(RGFW_window* window, Scene *scene);
	void render(RGFW_window *window);

	void add_shape(const Shape &shape);

	uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProps;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memProps);
		for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
			if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
				return i;
		}
		assert(false && "Failed to find memory type");
		return 0;
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
		VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
		bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &bi, nullptr, &buf);
		VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, buf, &mr);
		VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO }; ai.allocationSize = mr.size;
		VkMemoryAllocateFlagsInfo af{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT };
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ai.pNext = &af;
		ai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, props);
		vkAllocateMemory(device, &ai, nullptr, &mem);
		vkBindBufferMemory(device, buf, mem, 0);
	};

	VkCommandBuffer temp_command_buffer() {
		VkCommandBuffer cmd;
		VkCommandBufferAllocateInfo cba{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
		cba.commandPool = command_pool;
		cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cba.commandBufferCount = 1;
		vkAllocateCommandBuffers(device, &cba, &cmd);
		VkCommandBufferBeginInfo binfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		binfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(cmd, &binfo);
		return cmd;
	}
	void flush_and_destroy_command_buffer(VkCommandBuffer cmd) {
		vkEndCommandBuffer(cmd);
		VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
		vkQueueSubmit(graphics_queue, 1, &si, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphics_queue);
		vkFreeCommandBuffers(device, command_pool, 1, &cmd);
	}
};
