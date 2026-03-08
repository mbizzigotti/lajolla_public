#pragma once
#include "scene.h"
#define RGFW_IMPORT
#define RGFW_VULKAN
#define NOMINMAX
#include "3rdparty/RGFW.h"
#include "shaders/shared.slang"
#include <unordered_map>

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

struct VulkanRawBuffer {
	std::vector<uint8_t> data;
	VulkanBuffer buffer;
	VkDeviceAddress device_address;

	operator VkBuffer() { return buffer; }

	void Create(struct GPUDevice& gpu, VkFlags usage);
	void CreateFromStaging(struct GPUDevice& gpu, VkFlags usage);
	void GetDeviceAddress(VkDevice device);

	template <typename T>
	void Add(const T& t) {
		uint8_t* bytes = (uint8_t*)(&t);
		uint32_t size = sizeof(T);
		data.reserve(data.size() + size);
		for (uint32_t i = 0; i < size; ++i)
			data.emplace_back(bytes[i]);
	}

	template <typename T>
	void AddArray(const std::vector<T>& array) {
		uint8_t* bytes = (uint8_t*)(array.data());
		uint32_t size = array.size() * sizeof(T);
		data.reserve(data.size() + size);
		for (uint32_t i = 0; i < size; ++i)
			data.emplace_back(bytes[i]);
	}
};

struct ShaderParameterBlock {
	VkDescriptorSet        descriptor_set{ 0 };
	VkDescriptorSetLayout  descriptor_set_layout{ 0 };
	VkShaderStageFlags     shader_stages{ 0 };

	struct Descriptor
	{
		uint32_t         count;
		VkDescriptorType type;
	};

	std::vector<Descriptor> descriptors;
	std::unordered_map<std::string_view, uint32_t> descriptor_map;

	uint8_t write_memory[1024];
	uint32_t write_offset{ 0 };

	void add_binding(const char* name, VkDescriptorType type, uint32_t count = 1)
	{
		uint32_t index = (uint32_t)(descriptors.size());
		Descriptor desc = { .count = count, .type = type };
		descriptors.emplace_back(desc);
		descriptor_map[name] = index;
	}

	VkWriteDescriptorSet write(const char* name, VkAccelerationStructureKHR* as)
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
	
	VkWriteDescriptorSet write(const char* name, VkImageView image_view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL)
	{
		VkDescriptorImageInfo render_image_info = {
			.imageView = image_view,
			.imageLayout = layout,
		};
		assert(descriptor_map.contains(name));
		uint32_t binding = descriptor_map[name];
		assert(descriptors[binding].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
		return {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = binding,
			.descriptorCount = descriptors[binding].count,
			.descriptorType = descriptors[binding].type,
			.pImageInfo = push_write_structure(render_image_info),
		};
	}
	
	VkWriteDescriptorSet write(const char* name, VkBuffer buffer)
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

	template <typename T>
	T* push_write_structure(const T& s)
	{
		size_t offset = write_offset;
		void* ptr = write_memory + offset;
		write_offset += sizeof(T);
		memcpy(ptr, &s, sizeof(T));
		return (T*)(ptr);
	}

	void clear_write_memory() { write_offset = 0; }

	void CreateLayout(VkDevice device)
	{
		std::vector<VkDescriptorSetLayoutBinding> bindings;
		bindings.reserve(descriptors.size());
		for (uint32_t binding = 0; binding < descriptors.size(); ++binding) {
			bindings.emplace_back(VkDescriptorSetLayoutBinding{
				.binding = binding,
				.descriptorType = descriptors[binding].type,
				.descriptorCount = descriptors[binding].count,
				.stageFlags = shader_stages,
			});
		}
		VkDescriptorSetLayoutCreateInfo layout_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = (uint32_t)(bindings.size()),
			.pBindings = bindings.data(),
		};
		assert(vkCreateDescriptorSetLayout(device, &layout_info, 0, &descriptor_set_layout) == VK_SUCCESS);
	}

	void Allocate(VkDevice device, VkDescriptorPool pool)
	{
		VkDescriptorSetAllocateInfo allocate_info {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};
		assert(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set) == VK_SUCCESS);
	}

	void Destroy(VkDevice device)
	{
		if (descriptor_set_layout) vkDestroyDescriptorSetLayout(device, descriptor_set_layout, 0);
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
	VkPipelineLayout                pipeline_layout{ 0 };
	VkPipeline                      pipeline{ 0 };
	VulkanAccelerationStructure     tas{};
	VulkanBuffer                    sbt_buffer{ 0 };
	VkStridedDeviceAddressRegionKHR rgen_sbt{};
	VkStridedDeviceAddressRegionKHR miss_sbt{};
	VkStridedDeviceAddressRegionKHR chit_sbt{};
	VulkanRawBuffer                 instance_buffer{};
	VulkanRawBuffer                 info_buffer{};
	VulkanRawBuffer                 material_buffer{};
	VulkanRawBuffer                 shape_buffer{};
	VulkanRawBuffer                 vertex_buffer{};
	VulkanRawBuffer                 index_buffer{};
	VulkanRawBuffer                 uv_buffer{};
	VulkanRawBuffer                 normal_buffer{};

	ShaderParameterBlock            texture_block{};
	ShaderParameterBlock            scene_block{};
	
	std::vector<VulkanAccelerationStructure> bass;

	VkRenderPass render_pass{ 0 };
	VkFramebuffer frame_buffers[MAX_SWAP_CHAIN_IMAGES]{ 0 };

	GPUDevice();
	~GPUDevice();

	void attach(RGFW_window* window, Scene *scene);
	void render(RGFW_window *window);

	void add_shape_data(const Shape &shape);
	void add_shape(uint32_t index, const GPU::Shape& gpu_shape, const Shape &shape);
	VkPipelineShaderStageCreateInfo load_shader_stage(VkFlags stage, const char* name);

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
