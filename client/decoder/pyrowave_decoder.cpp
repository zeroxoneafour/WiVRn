/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Vaughan Milliman <me@vaughanm.xyz>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// PyroWave decoder, port of pyrowave_decoder.cpp from Granite to vulkan-hpp.
// See https://github.com/Themaister/pyrowave
//
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "pyrowave_decoder.h"

#include "application.h"
#include "scenes/stream.h"
#include "vk/specialization_constants.h"
#include "wivrn_shaders.h"

#include <format>
#include <map>
#include <ranges>
#include <spdlog/spdlog.h>

namespace
{
struct pyrowave_blit_handle : public wivrn::decoder::blit_handle
{
	std::atomic_bool & free;

	pyrowave_blit_handle(
	        const wivrn::from_headset::feedback & feedback,
	        const wivrn::to_headset::video_stream_data_shard::view_info_t & view_info,
	        vk::ImageView image_view,
	        vk::Image image,
	        vk::Extent2D extent,
	        vk::ImageLayout & current_layout,
	        vk::Semaphore semaphore,
	        uint64_t & semaphore_val,
	        std::atomic_bool & free) :
	        wivrn::decoder::blit_handle{feedback, view_info, image_view, image, extent, current_layout, semaphore, &semaphore_val},
	        free(free) {}
	~pyrowave_blit_handle()
	{
		free = true;
	}
};
} // namespace

namespace wivrn::pyrowave_core
{
namespace
{
struct dequantizer_push_data
{
	int32_t resolution[2];
	int32_t output_layer;
	int32_t block_offset_32x32;
	int32_t block_stride_32x32;
};

struct idwt_push_data
{
	int32_t resolution[2];
	float inv_resolution[2];
};

constexpr vk::SubgroupFeatureFlags required_ops =
        vk::SubgroupFeatureFlagBits::eVote |
        vk::SubgroupFeatureFlagBits::eBallot |
        vk::SubgroupFeatureFlagBits::eArithmetic |
        vk::SubgroupFeatureFlagBits::eShuffle |
        vk::SubgroupFeatureFlagBits::eShuffleRelative |
        vk::SubgroupFeatureFlagBits::eBasic;

bool prefer_texel_buffer(const device_caps & caps)
{
	// If the GPU is sufficiently competent with texel buffers, we can use that as a fallback to 8-bit storage.
	constexpr uint32_t vendor_amd = 0x1002;
	constexpr uint32_t vendor_nvidia = 0x10de;
	constexpr uint32_t vendor_intel = 0x8086;
	if (caps.max_texel_buffer_elements < 16 * 1024 * 1024)
		return false;
	return not(caps.storage_buffer_8bit and caps.storage_buffer_16bit) or
	       (caps.vendor_id != vendor_amd and caps.vendor_id != vendor_nvidia and caps.vendor_id != vendor_intel);
}

std::optional<device_caps::subgroup_config> dequant_subgroup(const device_caps & caps)
{
	if (auto config = caps.subgroup_for(4, 7, 128))
		return config;
	return caps.subgroup_for(2, 7, 128);
}

struct fragment_push_data
{
	float uv_offset[2];
	float half_texel_offset[2];
	float res_scale;
	int32_t aligned_transform_size;
};

// Intermediate precision of the iDWT, same as the FP16 storage of the wavelet bands
constexpr vk::Format fragment_format = vk::Format::eR16Sfloat;
constexpr vk::Format fragment_format_cbcr = vk::Format::eR16G16Sfloat;
constexpr vk::Format output_plane_format = vk::Format::eR8Unorm;

struct attachment
{
	vk::ImageView view;
	vk::Format format;
	vk::ImageLayout layout;
	vk::Extent2D extent;
};
} // namespace

// Direct port of the fragment path of PyroWave's decoder
struct decoder::fragment_path
{
	vk::raii::Device & device;
	vk::raii::ShaderModule vs = nullptr;
	std::array<vk::raii::ShaderModule, 3> fs{nullptr, nullptr, nullptr}; // CHROMA_CONFIG 0, 1, 2
	vk::raii::DescriptorSetLayout set_layout = nullptr;
	vk::raii::PipelineLayout layout = nullptr;

	struct level_t
	{
		// Output of the vertical passes: [even/odd pass][Y, CbCr]
		image_allocation vert[2][2];
		vk::raii::ImageView vert_views[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
		// Output of the horizontal pass: LL band of the previous level
		image_allocation horiz[num_components];
		vk::raii::ImageView horiz_views[num_components]{nullptr, nullptr, nullptr};

		// Input bands, the LL band is the output of the previous horizontal pass
		vk::raii::ImageView band_views[num_components][num_frequency_bands_per_level]{
		        {nullptr, nullptr, nullptr, nullptr},
		        {nullptr, nullptr, nullptr, nullptr},
		        {nullptr, nullptr, nullptr, nullptr},
		};
		vk::ImageView decoded[num_components][num_frequency_bands_per_level];
		vk::ImageLayout decoded_layout[num_components][num_frequency_bands_per_level];
	};
	std::array<level_t, decomposition_levels> levels;

	std::map<std::vector<std::pair<vk::Format, vk::ImageLayout>>, vk::raii::RenderPass> render_passes;
	std::map<std::pair<VkRenderPass, std::vector<VkImageView>>, std::pair<vk::raii::Framebuffer, vk::Extent2D>> framebuffers;
	std::map<std::tuple<VkRenderPass, int, bool, bool, bool, int>, vk::raii::Pipeline> pipelines;

	vk::raii::ShaderModule make_module(const shader_map & shaders, const char * name)
	{
		const auto & spirv = shaders.at(name);
		return vk::raii::ShaderModule(
		        device,
		        vk::ShaderModuleCreateInfo{
		                .codeSize = spirv.size() * sizeof(uint32_t),
		                .pCode = spirv.data(),
		        });
	}

	fragment_path(vk::raii::Device & device, const shader_map & shaders, const wavelet_buffers & buffers) :
	        device(device)
	{
		vs = make_module(shaders, "pyrowave_idwt_vs");
		fs[0] = make_module(shaders, "pyrowave_idwt_fs0");
		fs[1] = make_module(shaders, "pyrowave_idwt_fs1");
		fs[2] = make_module(shaders, "pyrowave_idwt_fs2");

		// Superset of the bindings of all chroma configurations
		std::array<vk::DescriptorSetLayoutBinding, 7> bindings;
		for (uint32_t i = 0; i < bindings.size(); ++i)
			bindings[i] = {
			        .binding = i,
			        .descriptorType = i == 2 ? vk::DescriptorType::eSampler : vk::DescriptorType::eSampledImage,
			        .descriptorCount = 1,
			        .stageFlags = vk::ShaderStageFlagBits::eFragment,
			};
		set_layout = vk::raii::DescriptorSetLayout(
		        device,
		        vk::DescriptorSetLayoutCreateInfo{
		                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
		                .bindingCount = uint32_t(bindings.size()),
		                .pBindings = bindings.data(),
		        });
		vk::PushConstantRange push_range{
		        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		        .size = sizeof(fragment_push_data),
		};
		layout = vk::raii::PipelineLayout(
		        device,
		        vk::PipelineLayoutCreateInfo{
		                .setLayoutCount = 1,
		                .pSetLayouts = &*set_layout,
		                .pushConstantRangeCount = 1,
		                .pPushConstantRanges = &push_range,
		        });

		for (int level = 0; level < decomposition_levels; level++)
		{
			auto & l = levels[level];
			vk::Extent2D horiz_extent{buffers.level_width(level), buffers.level_height(level)};
			vk::Extent2D vert_extent{horiz_extent.width, horiz_extent.height * 2};

			auto make_image = [&](vk::Extent2D extent, vk::Format format, const std::string & name) {
				return image_allocation(
				        device,
				        vk::ImageCreateInfo{
				                .imageType = vk::ImageType::e2D,
				                .format = format,
				                .extent = {extent.width, extent.height, 1},
				                .mipLevels = 1,
				                .arrayLayers = 1,
				                .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
				        },
				        {.usage = VMA_MEMORY_USAGE_AUTO},
				        name);
			};
			auto make_view = [&](vk::Image image, vk::Format format) {
				return vk::raii::ImageView(
				        device,
				        vk::ImageViewCreateInfo{
				                .image = image,
				                .viewType = vk::ImageViewType::e2D,
				                .format = format,
				                .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
				        });
			};

			for (int comp = 0; comp < num_components; comp++)
			{
				l.horiz[comp] = make_image(horiz_extent, fragment_format, std::format("pyrowave horizontal output (level {}, comp {})", level, comp));
				l.horiz_views[comp] = make_view(l.horiz[comp], fragment_format);
			}
			for (int pass = 0; pass < 2; pass++)
			{
				for (int comp = 0; comp < 2; comp++)
				{
					auto format = comp == 0 ? fragment_format : fragment_format_cbcr;
					l.vert[pass][comp] = make_image(vert_extent, format, std::format("pyrowave vertical {} input (level {}, comp {})", pass ? "odd" : "even", level, comp));
					l.vert_views[pass][comp] = make_view(l.vert[pass][comp], format);
				}
			}

			for (int comp = 0; comp < num_components; comp++)
			{
				for (int band = 0; band < num_frequency_bands_per_level; band++)
				{
					if (band == 0 and level < decomposition_levels - 1)
					{
						l.decoded[comp][band] = *l.horiz_views[comp];
						l.decoded_layout[comp][band] = vk::ImageLayout::eShaderReadOnlyOptimal;
						continue;
					}

					bool high_res = level < wavelet_fp16_levels;
					l.band_views[comp][band] = vk::raii::ImageView(
					        device,
					        vk::ImageViewCreateInfo{
					                .image = high_res ? vk::Image(buffers.wavelet_img_high_res) : vk::Image(buffers.wavelet_img_low_res),
					                .viewType = vk::ImageViewType::e2D,
					                .format = high_res ? vk::Format::eR16Sfloat : vk::Format::eR32Sfloat,
					                .subresourceRange = {
					                        .aspectMask = vk::ImageAspectFlagBits::eColor,
					                        .baseMipLevel = uint32_t(high_res ? level : level - wavelet_fp16_levels),
					                        .levelCount = 1,
					                        .baseArrayLayer = uint32_t(4 * comp + band),
					                        .layerCount = 1,
					                },
					        });
					l.decoded[comp][band] = *l.band_views[comp][band];
					l.decoded_layout[comp][band] = vk::ImageLayout::eGeneral;
				}
			}
		}
	}

	vk::RenderPass render_pass(std::span<const attachment> attachments)
	{
		std::vector<std::pair<vk::Format, vk::ImageLayout>> key;
		for (const auto & a: attachments)
			key.emplace_back(a.format, a.layout);

		auto it = render_passes.find(key);
		if (it != render_passes.end())
			return *it->second;

		std::vector<vk::AttachmentDescription> descriptions;
		std::vector<vk::AttachmentReference> references;
		for (const auto & a: attachments)
		{
			// Everything in the render area is written
			references.push_back({.attachment = uint32_t(descriptions.size()), .layout = a.layout});
			descriptions.push_back({
			        .format = a.format,
			        .samples = vk::SampleCountFlagBits::e1,
			        .loadOp = vk::AttachmentLoadOp::eDontCare,
			        .storeOp = vk::AttachmentStoreOp::eStore,
			        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
			        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
			        .initialLayout = a.layout,
			        .finalLayout = a.layout,
			});
		}
		vk::SubpassDescription subpass{
		        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
		        .colorAttachmentCount = uint32_t(references.size()),
		        .pColorAttachments = references.data(),
		};
		vk::raii::RenderPass rp(
		        device,
		        vk::RenderPassCreateInfo{
		                .attachmentCount = uint32_t(descriptions.size()),
		                .pAttachments = descriptions.data(),
		                .subpassCount = 1,
		                .pSubpasses = &subpass,
		        });
		return *render_passes.emplace(std::move(key), std::move(rp)).first->second;
	}

	// Begins a render pass on the attachments, returns the framebuffer size
	std::pair<vk::RenderPass, vk::Extent2D> begin(vk::raii::CommandBuffer & cmd, std::span<const attachment> attachments, std::optional<vk::Rect2D> render_area = std::nullopt)
	{
		auto rp = render_pass(attachments);

		std::pair<VkRenderPass, std::vector<VkImageView>> key{rp, {}};
		vk::Extent2D extent{UINT32_MAX, UINT32_MAX};
		for (const auto & a: attachments)
		{
			key.second.push_back(a.view);
			extent.width = std::min(extent.width, a.extent.width);
			extent.height = std::min(extent.height, a.extent.height);
		}

		auto it = framebuffers.find(key);
		if (it == framebuffers.end())
		{
			std::vector<vk::ImageView> views(attachments.size());
			std::ranges::transform(attachments, views.begin(), &attachment::view);
			vk::raii::Framebuffer fb(
			        device,
			        vk::FramebufferCreateInfo{
			                .renderPass = rp,
			                .attachmentCount = uint32_t(views.size()),
			                .pAttachments = views.data(),
			                .width = extent.width,
			                .height = extent.height,
			                .layers = 1,
			        });
			it = framebuffers.emplace(std::move(key), std::make_pair(std::move(fb), extent)).first;
		}

		cmd.beginRenderPass(
		        vk::RenderPassBeginInfo{
		                .renderPass = rp,
		                .framebuffer = *it->second.first,
		                .renderArea = render_area.value_or(vk::Rect2D{.extent = extent}),
		        },
		        vk::SubpassContents::eInline);
		return {rp, extent};
	}

	vk::Pipeline pipeline(vk::RenderPass rp, uint32_t color_count, int config, bool vertical, bool final_y, bool final_cbcr, int edge)
	{
		auto key = std::make_tuple(VkRenderPass(rp), config, vertical, final_y, final_cbcr, edge);
		auto it = pipelines.find(key);
		if (it != pipelines.end())
			return *it->second;

		auto spec = make_specialization_constants(VkBool32(vertical), VkBool32(final_y), VkBool32(final_cbcr), int32_t(edge));
		std::array stages{
		        vk::PipelineShaderStageCreateInfo{
		                .stage = vk::ShaderStageFlagBits::eVertex,
		                .module = *vs,
		                .pName = "main",
		                .pSpecializationInfo = spec,
		        },
		        vk::PipelineShaderStageCreateInfo{
		                .stage = vk::ShaderStageFlagBits::eFragment,
		                .module = *fs[config],
		                .pName = "main",
		                .pSpecializationInfo = spec,
		        },
		};
		vk::PipelineVertexInputStateCreateInfo vertex_input{};
		vk::PipelineInputAssemblyStateCreateInfo input_assembly{.topology = vk::PrimitiveTopology::eTriangleList};
		vk::PipelineViewportStateCreateInfo viewport{.viewportCount = 1, .scissorCount = 1};
		vk::PipelineRasterizationStateCreateInfo rasterization{
		        .polygonMode = vk::PolygonMode::eFill,
		        .cullMode = vk::CullModeFlagBits::eNone,
		        .lineWidth = 1,
		};
		vk::PipelineMultisampleStateCreateInfo multisample{.rasterizationSamples = vk::SampleCountFlagBits::e1};
		std::vector<vk::PipelineColorBlendAttachmentState> blend(
		        color_count,
		        {.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA});
		vk::PipelineColorBlendStateCreateInfo color_blend{
		        .attachmentCount = color_count,
		        .pAttachments = blend.data(),
		};
		std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamic{
		        .dynamicStateCount = uint32_t(dynamic_states.size()),
		        .pDynamicStates = dynamic_states.data(),
		};

		vk::raii::Pipeline p(
		        device,
		        nullptr,
		        vk::GraphicsPipelineCreateInfo{
		                .stageCount = uint32_t(stages.size()),
		                .pStages = stages.data(),
		                .pVertexInputState = &vertex_input,
		                .pInputAssemblyState = &input_assembly,
		                .pViewportState = &viewport,
		                .pRasterizationState = &rasterization,
		                .pMultisampleState = &multisample,
		                .pColorBlendState = &color_blend,
		                .pDynamicState = &dynamic,
		                .layout = *layout,
		                .renderPass = rp,
		        });
		return *pipelines.emplace(key, std::move(p)).first->second;
	}

	// Draws a full screen triangle in the scissor, skipped if empty
	void draw(vk::raii::CommandBuffer & cmd, vk::Pipeline pipeline, int32_t x, int32_t y, int32_t width, int32_t height)
	{
		if (width <= 0 or height <= 0)
			return;
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cmd.setScissor(0, vk::Rect2D{.offset = {x, y}, .extent = {uint32_t(width), uint32_t(height)}});
		cmd.draw(3, 1, 0, 0);
	}
};

bool decoder::prefers_fragment_path(const device_caps & caps)
{
	if (caps.driver_id == vk::DriverId{})
	{
		// Unknown driver, assume the proprietary one
		constexpr uint32_t vendor_qualcomm = 0x5143;
		constexpr uint32_t vendor_arm = 0x13b5;
		return caps.vendor_id == vendor_qualcomm or caps.vendor_id == vendor_arm;
	}

	switch (caps.driver_id)
	{
		// QCOM hardware struggles with compute in general and prefers fragment.
		// Turnip seems to like compute path just fine though ...
		case vk::DriverId::eQualcommProprietary:
			return true;

		// Mali heavily favors texture sampling over LS heavy content.
		case vk::DriverId::eArmProprietary:
		case vk::DriverId::eMesaPanvk:
			return true;

		default:
			return false;
	}
}

void decoder::check_support(const device_caps & caps)
{
	if (not caps.has_subgroup_ops(required_ops))
		throw std::runtime_error("pyrowave: missing required subgroup operations");
	if (not caps.push_descriptor)
		throw std::runtime_error("pyrowave: " VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME " not supported");
	if (not caps.storage_image_write_without_format)
		throw std::runtime_error("pyrowave: shaderStorageImageWriteWithoutFormat is required");
	if (not dequant_subgroup(caps))
		throw std::runtime_error("pyrowave: no compatible subgroup size");
	if (not(caps.storage_buffer_8bit and caps.storage_buffer_16bit) and not prefer_texel_buffer(caps))
		throw std::runtime_error("pyrowave: 8 and 16 bit storage or large texel buffers are required");
}

decoder::decoder(vk::raii::Device & device, const device_caps & caps, const shader_map & shaders, int width, int height, chroma_subsampling chroma, bool fragment_path) :
        wavelet_buffers(device, width, height, chroma),
        use_readonly_texel_buffer(prefer_texel_buffer(caps))
{
	check_support(caps);

	using type = vk::DescriptorType;
	if (use_readonly_texel_buffer)
		dequant_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_wavelet_dequant_texel",
		        {type::eStorageImage, type::eStorageBuffer, type::eUniformTexelBuffer, type::eUniformTexelBuffer, type::eUniformTexelBuffer},
		        sizeof(dequantizer_push_data),
		        *dequant_subgroup(caps));
	else
		dequant_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_wavelet_dequant",
		        {type::eStorageImage, type::eStorageBuffer, type::eStorageBuffer},
		        sizeof(dequantizer_push_data),
		        *dequant_subgroup(caps));

	if (fragment_path)
		fragment = std::make_unique<decoder::fragment_path>(device, shaders, *this);
	else
	{
		for (int dc_shift = 0; dc_shift < 2; ++dc_shift)
		{
			auto spec = make_specialization_constants(VkBool32(dc_shift));
			idwt_pipeline[dc_shift] = compute_pipeline(
			        device,
			        shaders,
			        caps.shader_float16 ? "pyrowave_idwt_fp16" : "pyrowave_idwt",
			        {type::eCombinedImageSampler, type::eStorageImage},
			        sizeof(idwt_push_data),
			        {},
			        spec);
		}
	}

	dequant_offset_buffer = buffer_allocation(
	        device,
	        vk::BufferCreateInfo{
	                .size = block_count_32x32 * sizeof(uint32_t),
	                .usage = vk::BufferUsageFlagBits::eStorageBuffer,
	        },
	        {
	                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
	                .usage = VMA_MEMORY_USAGE_AUTO,
	        },
	        "pyrowave dequant offsets");
	dequant_offset_buffer_cpu.resize(block_count_32x32);
	payload_data_cpu.reserve(1024 * 1024);

	clear();
}

decoder::~decoder() = default;

vk::ImageUsageFlags decoder::output_usage() const
{
	return fragment ? vk::ImageUsageFlagBits::eColorAttachment : vk::ImageUsageFlagBits::eStorage;
}

void decoder::clear()
{
	std::ranges::fill(dequant_offset_buffer_cpu, UINT32_MAX);
	decoded_blocks = 0;
	last_seq = UINT32_MAX;
	decoded_frame_for_current_sequence = false;
	total_blocks_in_sequence = block_count_32x32;
	payload_data_cpu.clear();
}

bool decoder::push_packet(std::span<const uint8_t> data)
{
	while (data.size() >= sizeof(bitstream_header))
	{
		bitstream_header header;
		memcpy(&header, data.data(), sizeof(header));

		if (header.extended != 0)
		{
			bitstream_sequence_header seq;
			memcpy(&seq, data.data(), sizeof(seq));

			if (seq.chroma_resolution != uint32_t(chroma))
			{
				spdlog::error("pyrowave: chroma resolution mismatch");
				return false;
			}

			uint8_t diff = (header.sequence - last_seq) & sequence_count_mask;
			if (last_seq != UINT32_MAX and diff > (sequence_count_mask / 2))
				return true;

			if (last_seq == UINT32_MAX or diff != 0)
			{
				clear();
				last_seq = header.sequence;
			}

			if (seq.code != bitstream_extended_code_start_of_frame)
			{
				spdlog::error("pyrowave: unrecognized sequence header mode {}", uint32_t(seq.code));
				return false;
			}

			if (int(seq.width_minus_1) + 1 != width or int(seq.height_minus_1) + 1 != height)
			{
				spdlog::error("pyrowave: dimension mismatch in sequence header, {}x{} != {}x{}",
				              seq.width_minus_1 + 1,
				              seq.height_minus_1 + 1,
				              width,
				              height);
				return false;
			}

			total_blocks_in_sequence = int(seq.total_blocks);
			data = data.subspan(sizeof(seq));
			continue;
		}

		size_t packet_size = header.payload_words * sizeof(uint32_t);
		if (packet_size > data.size())
		{
			spdlog::error("pyrowave: packet header states {} bytes, but only {} bytes left to parse", packet_size, data.size());
			return false;
		}

		bool restart;
		if (last_seq == UINT32_MAX)
		{
			restart = true;
		}
		else
		{
			uint8_t diff = (header.sequence - last_seq) & sequence_count_mask;
			if (diff > (sequence_count_mask / 2))
				return true;
			restart = diff != 0;
		}

		if (restart)
		{
			clear();
			last_seq = header.sequence;
		}

		if (header.block_index >= uint32_t(block_count_32x32))
		{
			spdlog::error("pyrowave: block_index {} is out of bounds (>= {})", uint32_t(header.block_index), block_count_32x32);
			return false;
		}

		if (not decode_packet(header, data.first(packet_size)))
			return false;

		data = data.subspan(packet_size);
	}

	if (not data.empty())
	{
		spdlog::error("pyrowave: did not consume packet completely");
		return false;
	}

	return true;
}

bool decoder::decode_is_ready(bool allow_partial_frame, int pristine_bands, float received_ratio) const
{
	if (decoded_frame_for_current_sequence)
		return false;

	if (last_seq == UINT32_MAX)
		return false;

	if (decoded_blocks < total_blocks_in_sequence)
	{
		if (not allow_partial_frame)
			return false;
		if (not has_pristine_bands(pristine_bands))
			return false;
		if (float(decoded_blocks) <= float(total_blocks_in_sequence) * received_ratio)
			return false;
	}

	return true;
}

void decoder::decode(vk::raii::CommandBuffer & cmd, const view_buffers & views)
{
	upload_payload();

	// Previous frame may still be reading the wavelet images
	memory_barrier(cmd,
	               vk::PipelineStageFlagBits::eComputeShader | (fragment ? vk::PipelineStageFlagBits::eFragmentShader : vk::PipelineStageFlags{}),
	               vk::AccessFlagBits::eShaderWrite,
	               vk::PipelineStageFlagBits::eComputeShader,
	               vk::AccessFlagBits::eShaderWrite);
	transition_wavelet_images(cmd);

	dequant(cmd);
	if (fragment)
		idwt_fragment(cmd, views);
	else
		idwt(cmd, views);

	decoded_frame_for_current_sequence = true;
}

bool decoder::decode_packet(const bitstream_header & header, std::span<const uint8_t> packet)
{
	auto & offset = dequant_offset_buffer_cpu[header.block_index];
	if (offset != UINT32_MAX)
		return true;

	if (sizeof(header) / sizeof(uint32_t) > header.payload_words)
	{
		spdlog::error("pyrowave: payload_words is not large enough");
		return false;
	}

	decoded_blocks++;
	offset = payload_data_cpu.size();

	size_t pos = payload_data_cpu.size();
	payload_data_cpu.resize(pos + header.payload_words);
	memcpy(payload_data_cpu.data() + pos, packet.data(), header.payload_words * sizeof(uint32_t));
	return true;
}

bool decoder::has_pristine_bands(int bands) const
{
	// Account for 4:2:0 where level0 will not have packets.
	// It's somewhat meaningless to ask for pristine bands all the way up to that point though.
	assert(bands < decomposition_levels);

	const auto block_is_missing = [&](uint32_t block_index) {
		return dequant_offset_buffer_cpu[block_index] == UINT32_MAX;
	};

	for (int band = 0; band < bands; band++)
	{
		for (auto & component: block_meta)
		{
			if (band == 0)
			{
				auto & meta = component[decomposition_levels - 1][0];
				for (int i = 0; i < meta.block_count_32x32; i++)
					if (block_is_missing(meta.block_offset_32x32 + i))
						return false;
			}
			else
			{
				// If we can reconstruct the LH, HL, HH bands, we can generate the higher-resolution LL band.
				for (int high_freq_bands = 1; high_freq_bands < 4; high_freq_bands++)
				{
					auto & meta = component[decomposition_levels - band][high_freq_bands];
					for (int i = 0; i < meta.block_count_32x32; i++)
						if (block_is_missing(meta.block_offset_32x32 + i))
							return false;
				}
			}
		}
	}

	return true;
}

void decoder::upload_payload()
{
	vk::DeviceSize required_size = payload_data_cpu.size() * sizeof(uint32_t);

	// Avoid edge case OOB access without robustness on the payload buffer during dequant.
	vk::DeviceSize required_size_padded = required_size + 16;

	if (not payload_data or required_size_padded > payload_data.info().size)
	{
		payload_views = {nullptr, nullptr, nullptr};
		payload_data = buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = std::max<vk::DeviceSize>(64 * 1024, required_size_padded * 2),
		                .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformTexelBuffer,
		        },
		        {
		                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		                .usage = VMA_MEMORY_USAGE_AUTO,
		        },
		        "pyrowave payload");

		if (use_readonly_texel_buffer)
		{
			const vk::Format formats[] = {vk::Format::eR32Uint, vk::Format::eR16Uint, vk::Format::eR8Uint};
			for (auto [view, format]: std::views::zip(payload_views, formats))
				view = vk::raii::BufferView(
				        *device,
				        vk::BufferViewCreateInfo{
				                .buffer = payload_data,
				                .format = format,
				                .range = vk::WholeSize,
				        });
		}
	}

	if (required_size)
		memcpy(payload_data.map(), payload_data_cpu.data(), required_size);
	vmaFlushAllocation(vk_allocator::instance(), payload_data, 0, VK_WHOLE_SIZE);

	memcpy(dequant_offset_buffer.map(), dequant_offset_buffer_cpu.data(), dequant_offset_buffer_cpu.size() * sizeof(uint32_t));
	vmaFlushAllocation(vk_allocator::instance(), dequant_offset_buffer, 0, VK_WHOLE_SIZE);
}

void decoder::dequant(vk::raii::CommandBuffer & cmd)
{
	dequant_pipeline.bind(cmd);

	for (int level = 0; level < decomposition_levels; level++)
	{
		for (int component = 0; component < num_components; component++)
		{
			if (skip(level, component, chroma))
				continue;

			if (use_readonly_texel_buffer)
				dequant_pipeline.push_descriptors(cmd,
				                                  {
				                                          descriptor::storage_image(layer_view(component, level)),
				                                          descriptor::storage_buffer(dequant_offset_buffer),
				                                          descriptor::uniform_texel_buffer(*payload_views[0]),
				                                          descriptor::uniform_texel_buffer(*payload_views[1]),
				                                          descriptor::uniform_texel_buffer(*payload_views[2]),
				                                  });
			else
				dequant_pipeline.push_descriptors(cmd,
				                                  {
				                                          descriptor::storage_image(layer_view(component, level)),
				                                          descriptor::storage_buffer(dequant_offset_buffer),
				                                          descriptor::storage_buffer(payload_data),
				                                  });

			for (int band = first_band(level); band < 4; band++)
			{
				dequantizer_push_data push{
				        .resolution = {int32_t(level_width(level)), int32_t(level_height(level))},
				        .output_layer = band,
				        .block_offset_32x32 = block_meta[component][level][band].block_offset_32x32,
				        .block_stride_32x32 = block_meta[component][level][band].block_stride_32x32,
				};
				dequant_pipeline.push_constants(cmd, push);
				cmd.dispatch((level_width(level) + 31) / 32, (level_height(level) + 31) / 32, 1);
			}
		}
	}

	memory_barrier(cmd,
	               vk::PipelineStageFlagBits::eComputeShader,
	               vk::AccessFlagBits::eShaderWrite,
	               fragment ? vk::PipelineStageFlagBits::eFragmentShader : vk::PipelineStageFlagBits::eComputeShader,
	               vk::AccessFlagBits::eShaderRead);
}

void decoder::idwt(vk::raii::CommandBuffer & cmd, const view_buffers & views)
{
	for (int input_level = decomposition_levels - 1; input_level >= 0; input_level--)
	{
		// Transposed.
		idwt_push_data push{
		        .resolution = {int32_t(level_height(input_level)), int32_t(level_width(input_level))},
		        .inv_resolution = {1.0f / float(level_height(input_level)), 1.0f / float(level_width(input_level))},
		};

		for (int c = 0; c < num_components; c++)
		{
			vk::ImageView output;
			bool dc_shift = false;
			if (input_level == 0)
			{
				if (c != 0 and chroma == chroma_subsampling::chroma_420)
					continue;
				output = views.planes[c];
				dc_shift = true;
			}
			else if (chroma == chroma_subsampling::chroma_420 and c != 0 and input_level == 1)
			{
				output = views.planes[c];
				dc_shift = true;
			}
			else
			{
				output = ll_view(c, input_level - 1);
			}

			auto & pipeline = idwt_pipeline[dc_shift];
			pipeline.bind(cmd);
			pipeline.push_constants(cmd, push);
			pipeline.push_descriptors(cmd,
			                          {
			                                  descriptor::sampled(layer_view(c, input_level), *mirror_repeat_sampler),
			                                  descriptor::storage_image(output),
			                          });
			cmd.dispatch((push.resolution[0] + 15) / 16, (push.resolution[1] + 15) / 16, 1);
		}

		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead);
	}
}

void decoder::idwt_fragment(vk::raii::CommandBuffer & cmd, const view_buffers & views)
{
	auto & f = *fragment;

	const auto layout_barrier = [](vk::Image image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags src, vk::AccessFlags dst) {
		return vk::ImageMemoryBarrier{
		        .srcAccessMask = src,
		        .dstAccessMask = dst,
		        .oldLayout = old_layout,
		        .newLayout = new_layout,
		        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
		        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
		        .image = image,
		        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
		};
	};
	const auto to_read_only = [&](std::span<image_allocation> images) {
		std::vector<vk::ImageMemoryBarrier> barriers;
		for (auto & image: images)
			barriers.push_back(layout_barrier(image, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead));
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barriers);
	};

	// Discard the intermediate images, previous frame may still be reading them
	{
		std::vector<vk::ImageMemoryBarrier> barriers;
		for (auto & level: f.levels)
		{
			for (auto & pass: level.vert)
				for (auto & image: pass)
					barriers.push_back(layout_barrier(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits::eColorAttachmentWrite));
			for (auto & image: level.horiz)
				barriers.push_back(layout_barrier(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {}, vk::AccessFlagBits::eColorAttachmentWrite));
		}
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {}, {}, barriers);
	}

	const auto sampled = [](vk::ImageView view, vk::ImageLayout layout) { return descriptor::sampled_image(view, layout); };
	constexpr auto read_only = vk::ImageLayout::eShaderReadOnlyOptimal;
	const vk::Sampler sampler = *mirror_repeat_sampler;

	for (int input_level = decomposition_levels - 1; input_level >= 0; input_level--)
	{
		const int output_level = input_level - 1;
		auto & l = f.levels[input_level];
		const bool has_chroma_output = output_level >= 0 or chroma == chroma_subsampling::chroma_444;

		// Vertical passes.
		for (int vert_pass = 0; vert_pass < 2; vert_pass++)
		{
			const vk::Extent2D vert_extent{level_width(input_level), level_height(input_level) * 2};
			std::vector<attachment> attachments{{*l.vert_views[vert_pass][0], fragment_format, vk::ImageLayout::eColorAttachmentOptimal, vert_extent}};
			if (has_chroma_output)
				attachments.push_back({*l.vert_views[vert_pass][1], fragment_format_cbcr, vk::ImageLayout::eColorAttachmentOptimal, vert_extent});

			auto [rp, extent] = f.begin(cmd, attachments);

			if (has_chroma_output)
				push_descriptors(cmd,
				                 vk::PipelineBindPoint::eGraphics,
				                 *f.layout,
				                 {
				                         sampled(l.decoded[0][vert_pass], l.decoded_layout[0][vert_pass]),
				                         sampled(l.decoded[0][vert_pass + 2], l.decoded_layout[0][vert_pass + 2]),
				                         descriptor::sampler(sampler),
				                         sampled(l.decoded[1][vert_pass], l.decoded_layout[1][vert_pass]),
				                         sampled(l.decoded[1][vert_pass + 2], l.decoded_layout[1][vert_pass + 2]),
				                         sampled(l.decoded[2][vert_pass], l.decoded_layout[2][vert_pass]),
				                         sampled(l.decoded[2][vert_pass + 2], l.decoded_layout[2][vert_pass + 2]),
				                 });
			else
				push_descriptors(cmd,
				                 vk::PipelineBindPoint::eGraphics,
				                 *f.layout,
				                 {
				                         sampled(l.decoded[0][vert_pass], l.decoded_layout[0][vert_pass]),
				                         sampled(l.decoded[0][vert_pass + 2], l.decoded_layout[0][vert_pass + 2]),
				                         descriptor::sampler(sampler),
				                 });

			const int32_t render_width = extent.width;
			const int32_t render_height = extent.height;

			// Set mirror point.
			// Work around broken Mali r38.1 compiler.
			// If it sees negative texture offsets it breaks the output for whatever reason (!?!?!?!).
			const float input_width = level_width(input_level);
			const float input_height = level_height(input_level);
			cmd.pushConstants<fragment_push_data>(
			        *f.layout,
			        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			        0,
			        fragment_push_data{
			                .uv_offset = {0, -2.0f / input_height},
			                .half_texel_offset = {0.5f / input_width, 0.5f / input_height},
			                .res_scale = float(render_height),
			                .aligned_transform_size = render_height,
			        });
			cmd.setViewport(0, vk::Viewport{0, 0, float(render_width), float(render_height), 0, 1});

			const int config = has_chroma_output ? 1 : 0;
			const uint32_t color_count = attachments.size();
			// Top edge condition, normal path, bottom edge condition
			f.draw(cmd, f.pipeline(rp, color_count, config, true, false, false, -1), 0, 0, render_width, 8);
			f.draw(cmd, f.pipeline(rp, color_count, config, true, false, false, 0), 0, 8, render_width, render_height - 16);
			f.draw(cmd, f.pipeline(rp, color_count, config, true, false, false, +1), 0, render_height - 8, render_width, 8);

			cmd.endRenderPass();
		}

		to_read_only(std::span(&l.vert[0][0], 4));

		// Horizontal pass
		std::vector<attachment> attachments;
		for (int comp = 0; comp < (has_chroma_output ? 3 : 1); comp++)
		{
			if (output_level < 0 or (output_level == 0 and chroma == chroma_subsampling::chroma_420 and comp != 0))
				attachments.push_back({views.planes[comp], output_plane_format, vk::ImageLayout::eGeneral, views.extents[comp]});
			else
				attachments.push_back({*f.levels[output_level].horiz_views[comp], fragment_format, vk::ImageLayout::eColorAttachmentOptimal, {level_width(output_level), level_height(output_level)}});
		}

		auto [rp, extent] = f.begin(cmd, attachments);

		const auto horizontal_descriptors = [&](bool with_chroma) {
			if (with_chroma)
				push_descriptors(cmd,
				                 vk::PipelineBindPoint::eGraphics,
				                 *f.layout,
				                 {
				                         sampled(*l.vert_views[0][0], read_only),
				                         sampled(*l.vert_views[1][0], read_only),
				                         descriptor::sampler(sampler),
				                         sampled(*l.vert_views[0][1], read_only),
				                         sampled(*l.vert_views[1][1], read_only),
				                 });
			else
				push_descriptors(cmd,
				                 vk::PipelineBindPoint::eGraphics,
				                 *f.layout,
				                 {
				                         sampled(*l.vert_views[0][0], read_only),
				                         sampled(*l.vert_views[1][0], read_only),
				                         descriptor::sampler(sampler),
				                 });
		};
		horizontal_descriptors(has_chroma_output);

		const int32_t aligned_render_width = aligned_width >> (output_level + 1);
		const int32_t aligned_render_height = aligned_height >> (output_level + 1);

		// Chroma output might be smaller than Y in output_level == 0 due to not using alignment.
		// This is reflected in the actual render area.
		const int32_t render_width = extent.width;
		const int32_t render_height = extent.height;

		// In case we're rendering to an output texture,
		// the render area might be smaller than we expect for purposes of alignment.
		// Use properly scaled viewport that we scissor away as needed.
		const vk::Viewport viewport{0, 0, float(aligned_render_width), float(aligned_render_height), 0, 1};
		cmd.setViewport(0, viewport);

		// Set mirror point.
		const float input_width = level_width(input_level);
		const float input_height = level_height(input_level) * 2;
		const fragment_push_data push{
		        .uv_offset = {-2.0f / input_width, 0},
		        .half_texel_offset = {0.5f / input_width, 0.5f / input_height},
		        .res_scale = float(aligned_render_width),
		        .aligned_transform_size = aligned_render_width,
		};
		cmd.pushConstants<fragment_push_data>(*f.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, push);

		const int config = has_chroma_output ? 2 : 0;
		const uint32_t color_count = attachments.size();
		const bool final_y = output_level < 0;
		const bool final_cbcr = output_level < 0 or (output_level == 0 and chroma == chroma_subsampling::chroma_420);

		// Left edge condition, normal condition, right edge condition
		f.draw(cmd, f.pipeline(rp, color_count, config, false, final_y, final_cbcr, -1), 0, 0, 8, render_height);
		f.draw(cmd, f.pipeline(rp, color_count, config, false, final_y, final_cbcr, 0), 8, 0, std::min(render_width - 8, aligned_render_width - 16), render_height);
		const int32_t aligned_x = aligned_render_width - 8;
		if (aligned_x < render_width)
			f.draw(cmd, f.pipeline(rp, color_count, config, false, final_y, final_cbcr, +1), aligned_x, 0, render_width - aligned_x, render_height);

		cmd.endRenderPass();

		// If chroma is subsampled, we cannot render the fully padded region in one render pass due to
		// rules regarding renderArea. renderArea cannot exceed the smallest image in the render pass.
		// We cannot use subpasses either, so split the render pass, but that's mostly fine,
		// since renderArea is non-overlapping.
		if (output_level == 0 and chroma == chroma_subsampling::chroma_420)
		{
			const auto & y = attachments[0];
			const auto & cb = attachments[1];
			const auto fixup = [&](vk::Rect2D area) {
				// Insert a simple by_region barrier to ensure we follow Vulkan rules for RW access.
				cmd.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    vk::PipelineStageFlagBits::eColorAttachmentOutput,
				                    vk::DependencyFlagBits::eByRegion,
				                    vk::MemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite, .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite},
				                    {},
				                    {});
				auto [rp, extent] = f.begin(cmd, std::span(&y, 1), area);
				horizontal_descriptors(false);
				cmd.setViewport(0, viewport);
				cmd.pushConstants<fragment_push_data>(*f.layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, push);
				// Always consider edge handling.
				f.draw(cmd, f.pipeline(rp, 1, 0, false, false, false, +1), area.offset.x, area.offset.y, area.extent.width, area.extent.height);
				cmd.endRenderPass();
			};

			// Need vertical fixup (very common for 1080p).
			if (cb.extent.height < y.extent.height)
				fixup({.offset = {0, int32_t(cb.extent.height)}, .extent = {y.extent.width, y.extent.height - cb.extent.height}});

			// Need horizontal fixup (very rare).
			if (cb.extent.width < y.extent.width)
				fixup({.offset = {int32_t(cb.extent.width), 0}, .extent = {y.extent.width - cb.extent.width, y.extent.height}});
		}

		if (output_level >= 0)
			to_read_only(f.levels[output_level].horiz);
	}

	// Avoid WAR hazard for dequantization.
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {}, {});
}
} // namespace wivrn::pyrowave_core

namespace wivrn
{
pyrowave_core::device_caps pyrowave_decoder::caps(vk::raii::PhysicalDevice & physical_device)
{
	// Only extensions are used to enable the features on the client
	return pyrowave_core::device_caps::query(physical_device, VK_API_VERSION_1_1, application::get_vk_device_extensions());
}

bool pyrowave_decoder::use_fragment_path(const pyrowave_core::device_caps & caps)
{
	if (const char * env = std::getenv("WIVRN_PYROWAVE_FRAGMENT"))
		return std::atoi(env);
	return pyrowave_core::decoder::prefers_fragment_path(caps);
}

pyrowave_decoder::image * pyrowave_decoder::get_free()
{
	for (auto & item: image_pool)
	{
		if (item.free.exchange(false))
			return &item;
	}
	return nullptr;
}

bool pyrowave_decoder::supported()
{
	try
	{
		auto & physical_device = application::get_physical_device();
		auto caps = pyrowave_decoder::caps(physical_device);
		pyrowave_core::decoder::check_support(caps);

		auto props = physical_device.getFormatProperties(output_format).optimalTilingFeatures;
		auto required = vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eCositedChromaSamples;
		if ((props & required) != required)
			throw std::runtime_error("pyrowave: 3-plane YCbCr 4:2:0 images are not supported");

		// Color attachment support is mandatory for R8
		auto plane_props = physical_device.getFormatProperties(vk::Format::eR8Unorm).optimalTilingFeatures;
		if (not use_fragment_path(caps) and not(plane_props & vk::FormatFeatureFlagBits::eStorageImage))
			throw std::runtime_error("pyrowave: R8 storage images are not supported");

		return true;
	}
	catch (std::exception & e)
	{
		spdlog::info("PyroWave decoder not supported: {}", e.what());
		return false;
	}
}

pyrowave_decoder::pyrowave_decoder(
        vk::raii::Device & device,
        vk::raii::PhysicalDevice & physical_device,
        uint32_t vk_queue_family_index,
        const wivrn::to_headset::video_stream_description & description,
        uint8_t stream_index,
        std::weak_ptr<scenes::stream> scene,
        shard_accumulator * accumulator) :
        device(device),
        stream_index(stream_index),
        extent{
                .width = description.width,
                .height = description.height / (stream_index == 2 ? 2u : 1u),
        },
        impl(device,
             caps(physical_device),
             ::shaders,
             extent.width,
             extent.height,
             pyrowave_core::chroma_subsampling::chroma_420,
             use_fragment_path(caps(physical_device))),
        command_pool(device,
                     vk::CommandPoolCreateInfo{
                             .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                             .queueFamilyIndex = vk_queue_family_index,
                     }),
        fence(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}),
        weak_scene(scene),
        accumulator(accumulator)
{
	cmd = std::move(device.allocateCommandBuffers({
	        .commandPool = *command_pool,
	        .commandBufferCount = 1,
	})[0]);

	// The alpha stream only uses the luma plane
	if (stream_index != 2)
		ycbcr_conversion = vk::raii::SamplerYcbcrConversion(
		        device,
		        {
		                .format = output_format,
		                .ycbcrModel = vk::SamplerYcbcrModelConversion::eYcbcr709,
		                .ycbcrRange = vk::SamplerYcbcrRange::eItuFull,
		                .chromaFilter = vk::Filter::eNearest,
		        });

	vk::StructureChain sampler_info{
	        vk::SamplerCreateInfo{
	                .magFilter = vk::Filter::eNearest,
	                .minFilter = vk::Filter::eNearest,
	                .mipmapMode = vk::SamplerMipmapMode::eNearest,
	                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
	                .addressModeW = vk::SamplerAddressMode::eClampToEdge,
	                .maxAnisotropy = 1,
	        },
	        vk::SamplerYcbcrConversionInfo{
	                .conversion = *ycbcr_conversion,
	        },
	};
	if (stream_index == 2)
		sampler_info.unlink<vk::SamplerYcbcrConversionInfo>();
	sampler_ = vk::raii::Sampler(device, sampler_info.get());

	spdlog::info("PyroWave decoder for stream {} uses the {} iDWT", stream_index, impl.output_usage() & vk::ImageUsageFlagBits::eColorAttachment ? "fragment" : "compute");

	for (auto & item: image_pool)
	{
		item.image = image_allocation(
		        device,
		        vk::ImageCreateInfo{
		                .flags = vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage,
		                .imageType = vk::ImageType::e2D,
		                .format = output_format,
		                .extent = {extent.width, extent.height, 1},
		                .mipLevels = 1,
		                .arrayLayers = 1,
		                .tiling = vk::ImageTiling::eOptimal,
		                .usage = vk::ImageUsageFlagBits::eSampled | impl.output_usage(),
		        },
		        {.usage = VMA_MEMORY_USAGE_AUTO},
		        "pyrowave image");

		vk::ImageViewUsageCreateInfo output_usage{.usage = impl.output_usage()};
		for (auto aspect: {vk::ImageAspectFlagBits::ePlane0, vk::ImageAspectFlagBits::ePlane1, vk::ImageAspectFlagBits::ePlane2})
			item.planes.emplace_back(
			        device,
			        vk::ImageViewCreateInfo{
			                .pNext = &output_usage,
			                .image = item.image,
			                .viewType = vk::ImageViewType::e2D,
			                .format = vk::Format::eR8Unorm,
			                .subresourceRange = {.aspectMask = aspect, .levelCount = 1, .layerCount = 1},
			        });

		vk::ImageViewUsageCreateInfo sampled_usage{.usage = vk::ImageUsageFlagBits::eSampled};
		vk::SamplerYcbcrConversionInfo conversion{
		        .pNext = &sampled_usage,
		        .conversion = *ycbcr_conversion,
		};
		item.view = vk::raii::ImageView(
		        device,
		        vk::ImageViewCreateInfo{
		                .pNext = stream_index == 2 ? (void *)&sampled_usage : (void *)&conversion,
		                .image = item.image,
		                .viewType = vk::ImageViewType::e2D,
		                .format = stream_index == 2 ? vk::Format::eR8Unorm : output_format,
		                .subresourceRange = {
		                        .aspectMask = stream_index == 2 ? vk::ImageAspectFlagBits::ePlane0 : vk::ImageAspectFlagBits::eColor,
		                        .levelCount = 1,
		                        .layerCount = 1,
		                },
		        });

		item.semaphore = vk::raii::Semaphore(
		        device,
		        vk::StructureChain{
		                vk::SemaphoreCreateInfo{},
		                vk::SemaphoreTypeCreateInfo{
		                        .semaphoreType = vk::SemaphoreType::eTimeline,
		                },
		        }
		                .get());
	}
}

pyrowave_decoder::~pyrowave_decoder()
{
	if (device.waitForFences(*fence, true, UINT64_MAX) != vk::Result::eSuccess)
		spdlog::warn("pyrowave: waitForFences failed");
}

void pyrowave_decoder::push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial)
{
	if (frame_index != current_frame)
	{
		bitstream.clear();
		current_frame = frame_index;
	}
	for (const auto & item: data)
		bitstream.insert(bitstream.end(), item.begin(), item.end());
}

void pyrowave_decoder::frame_completed(
        const from_headset::feedback & feedback,
        const to_headset::video_stream_data_shard::view_info_t & view_info)
{
	impl.clear();
	bool ok = impl.push_packet(bitstream);
	bitstream.clear();
	if (not ok or not impl.decode_is_ready(false))
	{
		spdlog::warn("pyrowave: invalid or incomplete frame, discard frame");
		return;
	}

	auto item = get_free();
	if (not item)
	{
		spdlog::warn("No image available in pool, discard frame");
		return;
	}

	auto handle = std::make_shared<pyrowave_blit_handle>(
	        feedback,
	        view_info,
	        *item->view,
	        item->image,
	        extent,
	        item->current_layout,
	        *item->semaphore,
	        item->semaphore_val,
	        item->free);

	if (device.waitForFences(*fence, true, UINT64_MAX) != vk::Result::eSuccess)
		spdlog::warn("waitForFences failed");

	cmd.reset();
	cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

	// Contents are entirely overwritten
	pyrowave_core::discard_to_general(cmd,
	                                  item->image,
	                                  vk::PipelineStageFlagBits::eAllCommands,
	                                  vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eColorAttachmentOutput,
	                                  vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eColorAttachmentWrite);
	item->current_layout = vk::ImageLayout::eGeneral;

	const vk::Extent2D chroma_extent{extent.width / 2, extent.height / 2};
	impl.decode(cmd,
	            {
	                    .planes = {*item->planes[0], *item->planes[1], *item->planes[2]},
	                    .extents = {extent, chroma_extent, chroma_extent},
	            });

	cmd.end();

	device.resetFences(*fence);
	application::get_queue().lock()->submit(
	        vk::StructureChain{
	                vk::SubmitInfo{
	                        .commandBufferCount = 1,
	                        .pCommandBuffers = &*cmd,
	                        .signalSemaphoreCount = 1,
	                        .pSignalSemaphores = &*item->semaphore,
	                },
	                vk::TimelineSemaphoreSubmitInfo{
	                        .signalSemaphoreValueCount = 1,
	                        .pSignalSemaphoreValues = &++item->semaphore_val,
	                },
	        }
	                .get(),
	        *fence);

	if (auto scene = weak_scene.lock())
		scene->push_blit_handle(accumulator, std::move(handle));
}
} // namespace wivrn
