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

// PyroWave decoder, port of pyrowave_decoder.{hpp,cpp} from Granite to vulkan-hpp.
// Only the compute path is implemented.
// See https://github.com/Themaister/pyrowave
//
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#pragma once

#include "decoder.h"

#include "application.h"
#include "scenes/stream.h"
#include "vk/pyrowave_common.h"
#include "vk/specialization_constants.h"
#include "wivrn_shaders.h"

#include <atomic>
#include <ranges>
#include <spdlog/spdlog.h>

namespace wivrn
{
namespace pyrowave_core
{
class decoder : public wavelet_buffers
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

	compute_pipeline dequant_pipeline;
	compute_pipeline idwt_pipeline[2]; // without and with DC shift

	// Written by the CPU, the caller guarantees the GPU is not using them
	buffer_allocation dequant_offset_buffer;
	buffer_allocation payload_data;
	std::array<vk::raii::BufferView, 3> payload_views{nullptr, nullptr, nullptr}; // u32, u16, u8
	bool use_readonly_texel_buffer;

	std::vector<uint32_t> dequant_offset_buffer_cpu;
	std::vector<uint32_t> payload_data_cpu;
	int decoded_blocks = 0;
	int total_blocks_in_sequence = 0;
	uint32_t last_seq = UINT32_MAX;
	bool decoded_frame_for_current_sequence = false;

	static bool prefer_texel_buffer(const device_caps & caps)
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

	static constexpr vk::SubgroupFeatureFlags required_ops =
	        vk::SubgroupFeatureFlagBits::eVote |
	        vk::SubgroupFeatureFlagBits::eBallot |
	        vk::SubgroupFeatureFlagBits::eArithmetic |
	        vk::SubgroupFeatureFlagBits::eShuffle |
	        vk::SubgroupFeatureFlagBits::eShuffleRelative |
	        vk::SubgroupFeatureFlagBits::eBasic;

	static std::optional<device_caps::subgroup_config> dequant_subgroup(const device_caps & caps)
	{
		if (auto config = caps.subgroup_for(4, 7, 128))
			return config;
		return caps.subgroup_for(2, 7, 128);
	}

public:
	static void check_support(const device_caps & caps)
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

	decoder(vk::raii::Device & device, const device_caps & caps, const shader_map & shaders, int width, int height, chroma_subsampling chroma) :
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

	void clear()
	{
		std::ranges::fill(dequant_offset_buffer_cpu, UINT32_MAX);
		decoded_blocks = 0;
		last_seq = UINT32_MAX;
		decoded_frame_for_current_sequence = false;
		total_blocks_in_sequence = block_count_32x32;
		payload_data_cpu.clear();
	}

	bool push_packet(std::span<const uint8_t> data)
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

	bool decode_is_ready(bool allow_partial_frame, int pristine_bands = 2, float received_ratio = 0.9f) const
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

	// Output planes must be in general layout and writable as storage images.
	// The previous decode must have completed.
	void decode(vk::raii::CommandBuffer & cmd, const view_buffers & views)
	{
		upload_payload();

		// Previous frame may still be reading the wavelet images
		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite);
		transition_wavelet_images(cmd);

		dequant(cmd);
		idwt(cmd, views);

		decoded_frame_for_current_sequence = true;
	}

private:
	bool decode_packet(const bitstream_header & header, std::span<const uint8_t> packet)
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

	bool has_pristine_bands(int bands) const
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

	void upload_payload()
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

	void dequant(vk::raii::CommandBuffer & cmd)
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
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead);
	}

	void idwt(vk::raii::CommandBuffer & cmd, const view_buffers & views)
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
};
} // namespace pyrowave_core

class pyrowave_decoder : public decoder
{
	static const int image_count = 5;
	static constexpr vk::Format output_format = vk::Format::eG8B8R83Plane420Unorm;

	struct image
	{
		image_allocation image;
		std::vector<vk::raii::ImageView> planes; // storage views
		vk::raii::ImageView view = nullptr;      // sampled view
		vk::ImageLayout current_layout = vk::ImageLayout::eUndefined;
		std::atomic_bool free = true;
		vk::raii::Semaphore semaphore = nullptr;
		uint64_t semaphore_val = 0;
	};

	struct blit_handle : public wivrn::decoder::blit_handle
	{
		std::atomic_bool & free;

		blit_handle(
		        const from_headset::feedback & feedback,
		        const to_headset::video_stream_data_shard::view_info_t & view_info,
		        pyrowave_decoder::image & item,
		        vk::Extent2D extent) :
		        wivrn::decoder::blit_handle{feedback, view_info, *item.view, item.image, extent, item.current_layout, *item.semaphore, &item.semaphore_val},
		        free(item.free) {}
		~blit_handle()
		{
			free = true;
		}
	};

	vk::raii::Device & device;
	const uint8_t stream_index;
	const vk::Extent2D extent;

	pyrowave_core::decoder impl;

	vk::raii::SamplerYcbcrConversion ycbcr_conversion = nullptr;
	vk::raii::Sampler sampler_ = nullptr;

	vk::raii::CommandPool command_pool;
	vk::raii::CommandBuffer cmd = nullptr;
	vk::raii::Fence fence;

	std::array<image, image_count> image_pool;

	std::weak_ptr<scenes::stream> weak_scene;
	shard_accumulator * accumulator;

	uint64_t current_frame = 0;
	std::vector<uint8_t> bitstream;

	static pyrowave_core::device_caps caps(vk::raii::PhysicalDevice & physical_device)
	{
		// Only extensions are used to enable the features on the client
		return pyrowave_core::device_caps::query(physical_device, VK_API_VERSION_1_1, application::get_vk_device_extensions());
	}

	image * get_free()
	{
		for (auto & item: image_pool)
		{
			if (item.free.exchange(false))
				return &item;
		}
		return nullptr;
	}

public:
	static bool supported()
	{
		try
		{
			auto & physical_device = application::get_physical_device();
			pyrowave_core::decoder::check_support(caps(physical_device));

			auto props = physical_device.getFormatProperties(output_format).optimalTilingFeatures;
			auto required = vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eCositedChromaSamples;
			if ((props & required) != required)
				throw std::runtime_error("pyrowave: 3-plane YCbCr 4:2:0 images are not supported");

			auto plane_props = physical_device.getFormatProperties(vk::Format::eR8Unorm).optimalTilingFeatures;
			if (not(plane_props & vk::FormatFeatureFlagBits::eStorageImage))
				throw std::runtime_error("pyrowave: R8 storage images are not supported");

			return true;
		}
		catch (std::exception & e)
		{
			spdlog::info("PyroWave decoder not supported: {}", e.what());
			return false;
		}
	}

	pyrowave_decoder(
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
	        impl(device, caps(physical_device), ::shaders, extent.width, extent.height, pyrowave_core::chroma_subsampling::chroma_420),
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
			                .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			        },
			        {.usage = VMA_MEMORY_USAGE_AUTO},
			        "pyrowave image");

			vk::ImageViewUsageCreateInfo storage_usage{.usage = vk::ImageUsageFlagBits::eStorage};
			for (auto aspect: {vk::ImageAspectFlagBits::ePlane0, vk::ImageAspectFlagBits::ePlane1, vk::ImageAspectFlagBits::ePlane2})
				item.planes.emplace_back(
				        device,
				        vk::ImageViewCreateInfo{
				                .pNext = &storage_usage,
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

	~pyrowave_decoder()
	{
		if (device.waitForFences(*fence, true, UINT64_MAX) != vk::Result::eSuccess)
			spdlog::warn("pyrowave: waitForFences failed");
	}

	void push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial) override
	{
		if (frame_index != current_frame)
		{
			bitstream.clear();
			current_frame = frame_index;
		}
		for (const auto & item: data)
			bitstream.insert(bitstream.end(), item.begin(), item.end());
	}

	void frame_completed(
	        const from_headset::feedback & feedback,
	        const to_headset::video_stream_data_shard::view_info_t & view_info) override
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

		auto handle = std::make_shared<blit_handle>(feedback, view_info, *item, extent);

		if (device.waitForFences(*fence, true, UINT64_MAX) != vk::Result::eSuccess)
			spdlog::warn("waitForFences failed");

		cmd.reset();
		cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		// Contents are entirely overwritten
		pyrowave_core::discard_to_general(cmd, item->image, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
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

	vk::Sampler sampler() override
	{
		return *sampler_;
	}
};

} // namespace wivrn
