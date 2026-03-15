#pragma once
#include "scene.h"
#define RGFW_IMPORT
#define RGFW_VULKAN
#define NOMINMAX
#include "3rdparty/RGFW.h"
#include "shaders/Shared.slang"
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
	uint32_t Count() {
		assert(data.size() % sizeof(T) == 0);
		return data.size() / sizeof(T);
	}

	template <typename T>
	void AddZeros(size_t count) {
		uint32_t size = count * sizeof(T);
		data.reserve(data.size() + size);
		for (uint32_t i = 0; i < size; ++i)
			data.emplace_back(0);
	}

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

	template <typename To, typename From>
	void AddArrayAndConvert(const std::vector<From>& array) {
		size_t offset = data.size();
		data.resize(offset + array.size() * sizeof(To));
		for (const From& item : array) {
			To converted = (To)(item);
			memcpy(data.data() + offset, &converted, sizeof(To));
			offset += sizeof(To);
		}
	}
};

struct VulkanImage {
	VkDeviceMemory  memory{ 0 };
	VkImage         image{ 0 };
	VkImageView     view{ 0 };
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

	VkWriteDescriptorSet write(const char* name, VkAccelerationStructureKHR* as);
	VkWriteDescriptorSet write(const char* name, VkImageView image_view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);
	VkWriteDescriptorSet write(const char* name, VkBuffer buffer);
	VkWriteDescriptorSet write(const char* name, VkSampler sampler);
	VkWriteDescriptorSet write_many(const char* name, VkDescriptorImageInfo* images, uint32_t count);

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

struct Tonemapper {
	VkPipelineLayout      pipeline_layout{ 0 };
	VkPipeline            pipeline{ 0 };
	ShaderParameterBlock  block{};
	float                 exposure{ 1.0f };

	void Create(struct GPUDevice& device);
	void Destroy(VkDevice device);

	void SetImages(VkDevice device, VkImageView input, VkImageView output) {
		VkWriteDescriptorSet writes[] = {
			block.write("in", input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			block.write("out", output),
		};
		vkUpdateDescriptorSets(device, (uint32_t)std::size(writes), writes, 0, 0);
		block.clear_write_memory();
	}

	void Tonemap(VkCommandBuffer cmd, uint32_t width, uint32_t height);
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
	Tonemapper                      tonemapper{};
	VkDeviceMemory                  storage_memory{ 0 };
	VkImage                         storage_image{ 0 };
	VkImageView                     storage_view{ 0 };
	VulkanBuffer                    staging_image_buffer{};
	float*                          storage_mapped{ 0 };
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
	VulkanRawBuffer                 aabb_buffer{};
	VulkanRawBuffer                 vertex_buffer{};
	VulkanRawBuffer                 index_buffer{};
	VulkanRawBuffer                 uv_buffer{};
	VulkanRawBuffer                 normal_buffer{};
	VulkanRawBuffer                 light_buffer{};
	VulkanRawBuffer                 texture_buffer{};
	VulkanRawBuffer                 dist_buffer{};
	VkSampler                       sampler{ 0 };
	ShaderParameterBlock            scene_block{};
	uint32_t                        target_sample_count{ 16 * 1024 };
	std::string                     scene_name;
	
	std::vector<VulkanImage>                 textures;
	std::vector<VulkanAccelerationStructure> bass;

	VkRenderPass render_pass{ 0 };
	VkFramebuffer frame_buffers[MAX_SWAP_CHAIN_IMAGES]{ 0 };

	GPUDevice();
	~GPUDevice();

	void attach(RGFW_window* window, Scene *scene, const std::string &path);
	void render(RGFW_window *window);

	void add_texture(const Mipmap3& texture);
	void add_shape_data(const Shape &shape);
	void add_shape(uint32_t index, const GPU::Shape& gpu_shape, const Shape &shape, uint32_t sphere_index);
	GPU::TableDist1D add_dist_1d(const TableDist1D& table);
	GPU::TableDist2D add_dist_2d(const TableDist2D& table);
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
		VkBufferCreateInfo buffer_info {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		assert(vkCreateBuffer(device, &buffer_info, nullptr, &buf) == VK_SUCCESS);
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

	uint32_t format_size(VkFormat format) {
		switch (format) {
		case VK_FORMAT_R32G32B32A32_SFLOAT: return sizeof(Vector4f);
		default: return 0;
		}
	}

	void write_image(VkImage dstImage, const Image3& source, VkFormat format, VkImageLayout outLayout, int level) {
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
};
