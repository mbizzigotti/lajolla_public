#pragma once
#include "scene.h"
#define RGFW_IMPORT
#define RGFW_VULKAN
#define NOMINMAX
#include "3rdparty/RGFW.h"

#define MAX_FRAMES_IN_FLIGHT  2
#define MAX_SWAP_CHAIN_IMAGES 4 /* For mobile devices, this would need to be higher.. */
struct GPUDevice {
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

	VkDescriptorSetLayout descriptor_set_layout;
	VkPipelineLayout      pipeline_layout;
	VkPipeline            pipeline;

	GPUDevice();

	void attach(RGFW_window *window);
	void render(RGFW_window *window, Scene& scene);
};
