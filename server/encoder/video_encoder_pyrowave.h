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

// PyroWave encoder, port of pyrowave_encoder.{hpp,cpp} from Granite to vulkan-hpp.
// See https://github.com/Themaister/pyrowave
//
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#pragma once

#include "encoder/encoder_settings.h"
#include "encoder/idr_handler.h"
#include "util/u_logging.h"
#include "utils/wivrn_vk_bundle.h"
#include "video_encoder.h"
#include "vk/pyrowave_common.h"
#include "vk/specialization_constants.h"
#include "wivrn-server_shaders.h"

#include <cmath>
#include <format>
#include <ranges>

namespace wivrn
{
namespace pyrowave_core
{
class encoder : public wavelet_buffers
{
	static constexpr int block_space_subdivision = 16;
	static constexpr int num_rdo_buckets = 128;
	static constexpr int rdo_bucket_offset = 64;

	struct rd_operation
	{
		int32_t quant;
		uint16_t block_offset;
		uint16_t block_saving;
	};

	struct dwt_push_data
	{
		uint32_t resolution[2];
		float inv_resolution[2];
		uint32_t aligned_resolution[2];
	};

	struct quantizer_push_data
	{
		int32_t resolution[2];
		int32_t resolution_8x8_blocks[2];
		float inv_resolution[2];
		float input_layer;
		float quant_resolution;
		int32_t block_offset;
		int32_t block_stride;
		float rdo_distortion_scale;
	};

	struct block_packing_push_data
	{
		int32_t resolution[2];
		int32_t resolution_32x32_blocks[2];
		int32_t resolution_8x8_blocks[2];
		uint32_t quant_resolution_code;
		uint32_t sequence_count;
		uint32_t block_offset_32x32;
		uint32_t block_stride_32x32;
		uint32_t block_offset_8x8;
		uint32_t block_stride_8x8;
	};

	struct analyze_rate_control_push_data
	{
		int32_t resolution[2];
		int32_t resolution_8x8_blocks[2];
		int32_t block_offset_8x8;
		int32_t block_stride_8x8;
		int32_t block_offset_32x32;
		int32_t block_stride_32x32;
		uint32_t total_wg_count;
		uint32_t num_blocks_aligned;
		uint32_t block_index_shamt;
	};

	struct resolve_push_data
	{
		uint32_t target_payload_size;
		uint32_t num_blocks_per_subdivision;
	};

	compute_pipeline dwt_pipeline[2]; // without and with DC shift
	compute_pipeline quant_pipeline;
	compute_pipeline analyze_pipeline;
	compute_pipeline analyze_finalize_pipeline;
	compute_pipeline resolve_pipeline;
	compute_pipeline packing_pipeline;

	buffer_allocation bucket_buffer, meta_buffer, block_stat_buffer, payload_data, quant_buffer;

	uint32_t sequence_count = 0;

	static int compute_block_count_per_subdivision(int num_blocks)
	{
		int per_subdivision = align(num_blocks, block_space_subdivision) / block_space_subdivision;
		return std::bit_ceil(uint32_t(per_subdivision));
	}

	static device_caps::subgroup_config require(std::optional<device_caps::subgroup_config> config)
	{
		if (not config)
			throw std::runtime_error("pyrowave: no compatible subgroup size configuration");
		return *config;
	}

	buffer_allocation make_buffer(vk::DeviceSize size, const char * name)
	{
		return buffer_allocation(
		        *device,
		        vk::BufferCreateInfo{
		                .size = size,
		                .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
		        },
		        {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE},
		        name);
	}

public:
	struct bitstream_buffers
	{
		vk::Buffer meta;
		vk::Buffer bitstream;
		vk::DeviceSize bitstream_size;
		size_t target_size;
	};

	static void check_support(const device_caps & caps)
	{
		const auto required_ops =
		        vk::SubgroupFeatureFlagBits::eArithmetic |
		        vk::SubgroupFeatureFlagBits::eShuffle |
		        vk::SubgroupFeatureFlagBits::eShuffleRelative |
		        vk::SubgroupFeatureFlagBits::eVote |
		        vk::SubgroupFeatureFlagBits::eBallot |
		        vk::SubgroupFeatureFlagBits::eClustered |
		        vk::SubgroupFeatureFlagBits::eBasic;
		if (not caps.has_subgroup_ops(required_ops))
			throw std::runtime_error("pyrowave: missing required subgroup operations");
		if (not caps.push_descriptor)
			throw std::runtime_error("pyrowave: " VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME " not supported");
		if (not(caps.storage_buffer_8bit and caps.storage_buffer_16bit))
			throw std::runtime_error("pyrowave: 8 and 16 bit storage buffer access are required");
		if (not(caps.shader_int16 and caps.shader_float16))
			throw std::runtime_error("pyrowave: shaderInt16 and shaderFloat16 are required");
		if (not caps.storage_image_write_without_format)
			throw std::runtime_error("pyrowave: shaderStorageImageWriteWithoutFormat is required");
		if (not caps.subgroup_for(4, 4, 16) and not caps.subgroup_for(5, 5, 32) and not caps.subgroup_for(6, 6, 64))
			throw std::runtime_error("pyrowave: no compatible subgroup size");
	}

	encoder(vk::raii::Device & device, const device_caps & caps, const shader_map & shaders, int width, int height, chroma_subsampling chroma) :
	        wavelet_buffers(device, width, height, chroma)
	{
		check_support(caps);

		using type = vk::DescriptorType;

		// Only need simple 2-lane swaps.
		auto dwt_subgroup = caps.subgroup_for(2, 7, 64).value_or(device_caps::subgroup_config{});
		for (int dc_shift = 0; dc_shift < 2; ++dc_shift)
		{
			auto spec = make_specialization_constants(VkBool32(dc_shift));
			dwt_pipeline[dc_shift] = compute_pipeline(
			        device,
			        shaders,
			        caps.shader_float16 ? "pyrowave_dwt_fp16" : "pyrowave_dwt",
			        {type::eCombinedImageSampler, type::eStorageImage},
			        sizeof(dwt_push_data),
			        dwt_subgroup,
			        spec);
		}

		quant_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_wavelet_quant",
		        {type::eCombinedImageSampler, type::eStorageBuffer, type::eStorageBuffer, type::eStorageBuffer},
		        sizeof(quantizer_push_data),
		        require(caps.subgroup_for(3, 7, 128)));

		analyze_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_analyze_rate_control",
		        {type::eStorageBuffer, type::eStorageBuffer},
		        sizeof(analyze_rate_control_push_data),
		        require(caps.subgroup_for(4, 6, 64)));

		analyze_finalize_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_analyze_rate_control_finalize",
		        {type::eStorageBuffer},
		        0,
		        {});

		// The workgroup size is a specialization constant equal to the subgroup size
		for (int log2: {6, 4, 5})
		{
			auto subgroup = caps.subgroup_for(log2, log2, 1u << log2);
			if (not subgroup)
				continue;
			auto spec = make_specialization_constants(uint32_t(1u << log2));
			resolve_pipeline = compute_pipeline(
			        device,
			        shaders,
			        "pyrowave_resolve_rate_control",
			        {type::eStorageBuffer, type::eStorageBuffer},
			        sizeof(resolve_push_data),
			        *subgroup,
			        spec);
			break;
		}

		packing_pipeline = compute_pipeline(
		        device,
		        shaders,
		        "pyrowave_block_packing",
		        {type::eStorageBuffer, type::eStorageBuffer, type::eStorageBuffer, type::eStorageBuffer, type::eStorageBuffer, type::eStorageBuffer},
		        sizeof(block_packing_push_data),
		        require(caps.subgroup_for(4, 6, 64)));

		block_stat_buffer = make_buffer(block_count_8x8 * sizeof(block_stats), "pyrowave block stats");
		meta_buffer = make_buffer(block_count_8x8 * sizeof(pyrowave_core::block_meta), "pyrowave meta");
		// Worst case estimate.
		payload_data = make_buffer(aligned_width * aligned_height * 2, "pyrowave payload");
		quant_buffer = make_buffer(block_count_32x32 * sizeof(uint32_t), "pyrowave quant");
		bucket_buffer = make_buffer(
		        rdo_bucket_offset +
		                num_rdo_buckets * block_space_subdivision * sizeof(uint32_t) +
		                num_rdo_buckets * compute_block_count_per_subdivision(block_count_32x32) * block_space_subdivision * sizeof(rd_operation),
		        "pyrowave buckets");
	}

	vk::DeviceSize meta_required_size() const
	{
		return block_count_32x32 * sizeof(bitstream_packet);
	}

	// Records the encode commands, input planes must be in general layout
	void encode(vk::raii::CommandBuffer & cmd, const view_buffers & views, const bitstream_buffers & buffers)
	{
		sequence_count = (sequence_count + 1) & sequence_count_mask;

		// Previous frame may still be reading any of the intermediate buffers
		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
		               vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
		               vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
		               vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite);
		transition_wavelet_images(cmd);

		cmd.fillBuffer(payload_data, 0, 2 * sizeof(uint32_t), 0);
		cmd.fillBuffer(bucket_buffer, 0, vk::WholeSize, 0);
		cmd.fillBuffer(quant_buffer, 0, vk::WholeSize, 0);

		dwt(cmd, views);

		// Don't need to read the payload offset counter until quantizer.
		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eTransfer,
		               vk::AccessFlagBits::eTransferWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

		quant(cmd);
		analyze_rdo(cmd);
		resolve_rdo(cmd, buffers.target_size);
		block_packing(cmd, buffers);
	}

	// Build the frame from the block packets written by encode, returns false on error
	bool packetize(std::vector<uint8_t> & out, std::span<const bitstream_packet> meta, std::span<const uint32_t> bitstream) const
	{
		out.clear();
		if (meta.size() < size_t(block_count_32x32))
			return false;

		bitstream_sequence_header header{
		        .width_minus_1 = uint32_t(width - 1),
		        .height_minus_1 = uint32_t(height - 1),
		        .extended = 1,
		        .code = bitstream_extended_code_start_of_frame,
		        .chroma_resolution = uint32_t(chroma),
		};

		size_t total_size = sizeof(header);
		for (const auto & packet: meta.first(block_count_32x32))
		{
			if (packet.num_words == 0)
				continue;
			if (size_t(packet.offset_u32) + packet.num_words > bitstream.size())
			{
				U_LOG_E("pyrowave: block out of bitstream bounds");
				return false;
			}
			if (header.total_blocks == 0)
				header.sequence = reinterpret_cast<const bitstream_header *>(&bitstream[packet.offset_u32])->sequence;
			header.total_blocks++;
			total_size += packet.num_words * sizeof(uint32_t);
		}

		out.resize(total_size);
		auto * pos = out.data();
		memcpy(pos, &header, sizeof(header));
		pos += sizeof(header);
		for (const auto & packet: meta.first(block_count_32x32))
		{
			if (packet.num_words == 0)
				continue;
			memcpy(pos, &bitstream[packet.offset_u32], packet.num_words * sizeof(uint32_t));
			pos += packet.num_words * sizeof(uint32_t);
		}
		return true;
	}

private:
	float get_quant_rdo_distortion_scale(int level, int component, int band) const
	{
		// From my Linelet master thesis. Copy paste 11 years later, ah yes :D
		float horiz_midpoint = (band & 1) ? 0.75f : 0.25f;
		float vert_midpoint = (band & 2) ? 0.75f : 0.25f;

		// Normal PC monitors.
		constexpr float dpi = 96.0f;
		// Compromise between couch gaming and desktop.
		constexpr float viewing_distance = 1.0f;
		constexpr float cpd_nyquist = 0.34f * viewing_distance * dpi;

		float cpd = std::sqrt(horiz_midpoint * horiz_midpoint + vert_midpoint * vert_midpoint) *
		            cpd_nyquist * std::exp2(-float(level));

		// Don't allow a situation where we're quantizing LL band hard.
		cpd = std::max(cpd, 8.0f);

		float csf = 2.6f * (0.0192f + 0.114f * cpd) * std::exp(-std::pow(0.114f * cpd, 1.1f));

		// Heavily discount chroma quality.
		if (component != 0 and level != decomposition_levels - 1 and chroma == chroma_subsampling::chroma_420)
			csf *= 0.6f;

		// Due to filtering, distortion in lower bands will result in more noise power.
		// By scaling the distortion by this factor, we ensure uniform results.
		float resolution = get_noise_power_normalized_quant_resolution(level, component, band);
		float weighted_resolution = csf * resolution;

		// The distortion is scaled in terms of power, not amplitude.
		return weighted_resolution * weighted_resolution;
	}

	float get_quant_resolution(int level, int component, int band) const
	{
		// FP16 range is limited, and this is more than a good enough initial estimate.
		return std::min<float>(4096.0f, get_noise_power_normalized_quant_resolution(level, component, band));
	}

	float get_noise_power_normalized_quant_resolution(int level, int component, int band) const
	{
		// The initial quantization resolution aims for a flat spectrum with noise power normalization.
		// The low-pass gain for CDF 9/7 is 6 dB (1 bit). Every decomposition level subtracts 6 dB.
		int bits = 8;

		if (band == 0)
			bits += 2;
		else if (band < 3)
			bits += 1;

		bits += level;

		// Chroma starts at level 1, subtract one bit.
		if (component != 0)
			bits--;

		return float(1 << bits);
	}

	void dwt(vk::raii::CommandBuffer & cmd, const view_buffers & views)
	{
		for (int output_level = 0; output_level < decomposition_levels; output_level++)
		{
			dwt_push_data push{};
			if (output_level > 0)
			{
				push.resolution[0] = push.aligned_resolution[0] = level_width(output_level - 1);
				push.resolution[1] = push.aligned_resolution[1] = level_height(output_level - 1);
			}
			else
			{
				push.resolution[0] = views.extents[0].width;
				push.resolution[1] = views.extents[0].height;
				push.aligned_resolution[0] = aligned_width;
				push.aligned_resolution[1] = aligned_height;
			}

			for (int c = 0; c < num_components; c++)
			{
				vk::ImageView input;
				dwt_push_data component_push = push;
				bool dc_shift = false;

				if (output_level == 0)
				{
					if (c != 0 and chroma == chroma_subsampling::chroma_420)
						continue;
					dc_shift = true;
					input = views.planes[c];
				}
				else if (chroma == chroma_subsampling::chroma_420 and c != 0 and output_level == 1)
				{
					component_push.resolution[0] = views.extents[c].width;
					component_push.resolution[1] = views.extents[c].height;
					component_push.aligned_resolution[0] = aligned_width >> output_level;
					component_push.aligned_resolution[1] = aligned_height >> output_level;
					dc_shift = true;
					input = views.planes[c];
				}
				else
				{
					input = ll_view(c, output_level - 1);
				}
				component_push.inv_resolution[0] = 1.0f / float(component_push.resolution[0]);
				component_push.inv_resolution[1] = 1.0f / float(component_push.resolution[1]);

				auto & pipeline = dwt_pipeline[dc_shift];
				pipeline.bind(cmd);
				pipeline.push_constants(cmd, component_push);
				pipeline.push_descriptors(cmd,
				                          {
				                                  descriptor::sampled(input, *mirror_repeat_sampler),
				                                  descriptor::storage_image(layer_view(c, output_level)),
				                          });
				cmd.dispatch((component_push.aligned_resolution[0] + 31) / 32, (component_push.aligned_resolution[1] + 31) / 32, 1);
			}

			memory_barrier(cmd,
			               vk::PipelineStageFlagBits::eComputeShader,
			               vk::AccessFlagBits::eShaderWrite,
			               vk::PipelineStageFlagBits::eComputeShader,
			               vk::AccessFlagBits::eShaderRead);
		}
	}

	void quant(vk::raii::CommandBuffer & cmd)
	{
		quant_pipeline.bind(cmd);

		for (int level = 0; level < decomposition_levels; level++)
		{
			for (int component = 0; component < num_components; component++)
			{
				if (skip(level, component, chroma))
					continue;

				quant_pipeline.push_descriptors(cmd,
				                                {
				                                        descriptor::sampled(layer_view(component, level), *border_sampler),
				                                        descriptor::storage_buffer(meta_buffer),
				                                        descriptor::storage_buffer(block_stat_buffer),
				                                        descriptor::storage_buffer(payload_data),
				                                });

				for (int band = first_band(level); band < 4; band++)
				{
					float quant_res = get_quant_resolution(level, component, band);

					quantizer_push_data push{
					        .resolution = {int32_t(level_width(level)), int32_t(level_height(level))},
					        .resolution_8x8_blocks = {int32_t(level_width(level) + 7) / 8, int32_t(level_height(level) + 7) / 8},
					        .inv_resolution = {1.0f / float(level_width(level)), 1.0f / float(level_height(level))},
					        .input_layer = float(band),
					        .quant_resolution = 1.0f / decode_quant(encode_quant(1.0f / quant_res)),
					        .block_offset = block_meta[component][level][band].block_offset_8x8,
					        .block_stride = block_meta[component][level][band].block_stride_8x8,
					        .rdo_distortion_scale = get_quant_rdo_distortion_scale(level, component, band) * (1.0f / 256.0f),
					};
					quant_pipeline.push_constants(cmd, push);
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

	void analyze_rdo(vk::raii::CommandBuffer & cmd)
	{
		analyze_pipeline.bind(cmd);
		analyze_pipeline.push_descriptors(cmd,
		                                  {
		                                          descriptor::storage_buffer(bucket_buffer),
		                                          descriptor::storage_buffer(block_stat_buffer),
		                                  });

		int per_subdivision = compute_block_count_per_subdivision(block_count_32x32);

		for (int level = 0; level < decomposition_levels; level++)
		{
			for (int component = 0; component < num_components; component++)
			{
				if (skip(level, component, chroma))
					continue;

				for (int band = first_band(level); band < 4; band++)
				{
					const auto & meta = block_meta[component][level][band];
					analyze_rate_control_push_data push{
					        .resolution = {int32_t(level_width(level)), int32_t(level_height(level))},
					        .resolution_8x8_blocks = {int32_t(level_width(level) + 7) / 8, int32_t(level_height(level) + 7) / 8},
					        .block_offset_8x8 = meta.block_offset_8x8,
					        .block_stride_8x8 = meta.block_stride_8x8,
					        .block_offset_32x32 = meta.block_offset_32x32,
					        .block_stride_32x32 = meta.block_stride_32x32,
					        .total_wg_count = uint32_t(block_count_32x32),
					        .num_blocks_aligned = uint32_t(per_subdivision * block_space_subdivision),
					        .block_index_shamt = uint32_t(std::countr_zero(uint32_t(per_subdivision))),
					};
					analyze_pipeline.push_constants(cmd, push);
					cmd.dispatch((level_width(level) + 31) / 32, (level_height(level) + 31) / 32, 1);
				}
			}
		}

		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);

		analyze_finalize_pipeline.bind(cmd);
		analyze_finalize_pipeline.push_descriptors(cmd, {descriptor::storage_buffer(bucket_buffer)});
		cmd.dispatch(1, 1, 1);

		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead);
	}

	void resolve_rdo(vk::raii::CommandBuffer & cmd, size_t target_payload_size)
	{
		if (target_payload_size >= sizeof(bitstream_sequence_header))
			target_payload_size -= sizeof(bitstream_sequence_header);

		resolve_pipeline.bind(cmd);
		resolve_pipeline.push_constants(cmd,
		                                resolve_push_data{
		                                        .target_payload_size = uint32_t(target_payload_size / sizeof(uint32_t)),
		                                        .num_blocks_per_subdivision = uint32_t(compute_block_count_per_subdivision(block_count_32x32)),
		                                });
		resolve_pipeline.push_descriptors(cmd,
		                                  {
		                                          descriptor::storage_buffer(bucket_buffer),
		                                          descriptor::storage_buffer(quant_buffer),
		                                  });
		cmd.dispatch(num_rdo_buckets * block_space_subdivision, 1, 1);

		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderRead);
	}

	void block_packing(vk::raii::CommandBuffer & cmd, const bitstream_buffers & buffers)
	{
		packing_pipeline.bind(cmd);
		packing_pipeline.push_descriptors(cmd,
		                                  {
		                                          descriptor::storage_buffer(buffers.bitstream, 0, buffers.bitstream_size),
		                                          descriptor::storage_buffer(buffers.meta, 0, meta_required_size()),
		                                          descriptor::storage_buffer(meta_buffer),
		                                          descriptor::storage_buffer(payload_data),
		                                          descriptor::storage_buffer(block_stat_buffer),
		                                          descriptor::storage_buffer(quant_buffer),
		                                  });

		for (int level = 0; level < decomposition_levels; level++)
		{
			for (int component = 0; component < num_components; component++)
			{
				if (skip(level, component, chroma))
					continue;

				for (int band = first_band(level); band < 4; band++)
				{
					const auto & meta = block_meta[component][level][band];
					block_packing_push_data push{
					        .resolution = {int32_t(level_width(level)), int32_t(level_height(level))},
					        .resolution_32x32_blocks = {int32_t(level_width(level) + 31) / 32, int32_t(level_height(level) + 31) / 32},
					        .resolution_8x8_blocks = {int32_t(level_width(level) + 7) / 8, int32_t(level_height(level) + 7) / 8},
					        .quant_resolution_code = encode_quant(1.0f / get_quant_resolution(level, component, band)),
					        .sequence_count = sequence_count,
					        .block_offset_32x32 = uint32_t(meta.block_offset_32x32),
					        .block_stride_32x32 = uint32_t(meta.block_stride_32x32),
					        .block_offset_8x8 = uint32_t(meta.block_offset_8x8),
					        .block_stride_8x8 = uint32_t(meta.block_stride_8x8),
					};
					packing_pipeline.push_constants(cmd, push);
					cmd.dispatch((push.resolution_32x32_blocks[0] + 1) / 2, (push.resolution_32x32_blocks[1] + 1) / 2, 1);
				}
			}
		}

		memory_barrier(cmd,
		               vk::PipelineStageFlagBits::eComputeShader,
		               vk::AccessFlagBits::eShaderWrite,
		               vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
		               vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
	}
};
} // namespace pyrowave_core

class video_encoder_pyrowave : public video_encoder
{
	class dummy_idr_handler : public idr_handler
	{
	public:
		void on_feedback(const from_headset::feedback &) override {}
		void reset() override {}
		bool should_skip(uint64_t) override
		{
			return false;
		}
	};

	vk_bundle & vk;
	pyrowave_core::encoder encoder;
	vk::raii::CommandPool cmd_pool;
	vk::Format plane_formats[2];

	uint64_t bitrate;
	float fps;

	struct slot_t
	{
		vk::raii::CommandBuffer cmd = nullptr;
		vk::raii::Fence fence = nullptr;
		buffer_allocation meta, bitstream;           // written by the GPU
		buffer_allocation host_meta, host_bitstream; // read by the CPU
		std::vector<uint8_t> output;
	};
	std::array<slot_t, num_slots> slots;

	// Plane views of the images from the compositor
	struct input_t
	{
		vk::Image image;
		std::vector<vk::raii::ImageView> views;
	};
	std::vector<input_t> inputs;

	// The alpha stream is in the top half of the luma plane of the compositor image, it is copied
	// so that the sampled image has the size of the stream.
	image_allocation alpha_image;
	std::vector<vk::raii::ImageView> alpha_views; // luma, constant chroma

	buffer_allocation make_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage, bool host, const std::string & name)
	{
		return buffer_allocation(
		        vk.device,
		        vk::BufferCreateInfo{
		                .size = size,
		                .usage = usage,
		        },
		        {
		                .flags = host ? VmaAllocationCreateFlags(VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT) : 0,
		                .usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		        },
		        name);
	}

	pyrowave_core::view_buffers input_views(vk::raii::CommandBuffer & cmd, vk::Image image)
	{
		const vk::Extent2D chroma_extent{extent.width / 2, extent.height / 2};

		if (stream_idx == 2)
		{
			// Previous frame may still be sampling it
			pyrowave_core::discard_to_general(cmd, alpha_image, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
			cmd.copyImage(image,
			              vk::ImageLayout::eGeneral,
			              alpha_image,
			              vk::ImageLayout::eGeneral,
			              vk::ImageCopy{
			                      .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::ePlane0, .baseArrayLayer = stream_idx, .layerCount = 1},
			                      .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
			                      .extent = {extent.width, extent.height, 1},
			              });
			pyrowave_core::memory_barrier(cmd, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
			return {
			        .planes = {*alpha_views[0], *alpha_views[1], *alpha_views[1]},
			        .extents = {extent, chroma_extent, chroma_extent},
			};
		}

		auto it = std::ranges::find(inputs, image, &input_t::image);
		if (it == inputs.end())
		{
			vk::ImageViewUsageCreateInfo usage{
			        .usage = vk::ImageUsageFlagBits::eSampled,
			};
			const vk::ComponentSwizzle chroma[] = {
			        vk::ComponentSwizzle::eR, // Cb
			        vk::ComponentSwizzle::eG, // Cr
			};
			input_t input{.image = image};
			for (int plane = 0; plane < 3; ++plane)
			{
				vk::ImageViewCreateInfo info{
				        .pNext = &usage,
				        .image = image,
				        .viewType = vk::ImageViewType::e2D,
				        .format = plane_formats[plane > 0],
				        .subresourceRange = {
				                .aspectMask = plane == 0 ? vk::ImageAspectFlagBits::ePlane0 : vk::ImageAspectFlagBits::ePlane1,
				                .levelCount = 1,
				                .baseArrayLayer = stream_idx,
				                .layerCount = 1,
				        },
				};
				if (plane > 0)
					info.components = {chroma[plane - 1], chroma[plane - 1], chroma[plane - 1], chroma[plane - 1]};
				input.views.emplace_back(vk.device, info);
			}
			inputs.push_back(std::move(input));
			it = inputs.end() - 1;
		}

		return {
		        .planes = {*it->views[0], *it->views[1], *it->views[2]},
		        .extents = {extent, chroma_extent, chroma_extent},
		};
	}

	size_t target_size() const
	{
		auto size = uint64_t(bitrate / fps / 8);
		// Keep the size reasonable: bitstream alignment and at least the sequence header
		return std::max<uint64_t>(size & ~uint64_t(3), 64);
	}

	void ensure_buffers(slot_t & slot, size_t target)
	{
		vk::DeviceSize bitstream_size = target + 2 * encoder.meta_required_size();
		if (slot.bitstream and slot.bitstream.info().size >= bitstream_size)
			return;

		slot.bitstream = make_buffer(bitstream_size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, false, std::format("pyrowave bitstream {}", stream_idx));
		slot.host_bitstream = make_buffer(bitstream_size, vk::BufferUsageFlagBits::eTransferDst, true, std::format("pyrowave host bitstream {}", stream_idx));
	}

public:
	static bool supported(vk_bundle & vk)
	{
		try
		{
			pyrowave_core::encoder::check_support(pyrowave_core::device_caps::query(vk.physical_device, vk_bundle::api_version, vk.device_extensions));
			return true;
		}
		catch (std::exception & e)
		{
			U_LOG_W("%s", e.what());
			return false;
		}
	}

	video_encoder_pyrowave(vk_bundle & vk, const encoder_settings & settings, uint8_t stream_idx) :
	        video_encoder(vk, stream_idx, vk.queue.family_index, settings, std::make_unique<dummy_idr_handler>(), true),
	        vk(vk),
	        encoder(vk.device,
	                pyrowave_core::device_caps::query(vk.physical_device, vk_bundle::api_version, vk.device_extensions),
	                ::shaders,
	                settings.width,
	                settings.height,
	                pyrowave_core::chroma_subsampling::chroma_420),
	        cmd_pool(vk.device,
	                 vk::CommandPoolCreateInfo{
	                         .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
	                         .queueFamilyIndex = vk.queue.family_index,
	                 }),
	        bitrate(settings.bitrate),
	        fps(settings.fps)
	{
		switch (settings.bit_depth)
		{
			case 8:
				plane_formats[0] = vk::Format::eR8Unorm;
				plane_formats[1] = vk::Format::eR8G8Unorm;
				break;
			case 10:
				plane_formats[0] = vk::Format::eR16Unorm;
				plane_formats[1] = vk::Format::eR16G16Unorm;
				break;
			default:
				throw std::runtime_error("pyrowave: unsupported bit depth " + std::to_string(settings.bit_depth));
		}

		vk.name(cmd_pool, std::format("pyrowave encoder {} command pool", stream_idx));

		if (stream_idx == 2)
		{
			alpha_image = image_allocation(
			        vk.device,
			        vk::ImageCreateInfo{
			                .imageType = vk::ImageType::e2D,
			                .format = plane_formats[0],
			                .extent = {extent.width, extent.height, 1},
			                .mipLevels = 1,
			                .arrayLayers = 1,
			                .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			        },
			        {.usage = VMA_MEMORY_USAGE_AUTO},
			        "pyrowave alpha");
			for (auto swizzle: {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eZero})
				alpha_views.emplace_back(
				        vk.device,
				        vk::ImageViewCreateInfo{
				                .image = alpha_image,
				                .viewType = vk::ImageViewType::e2D,
				                .format = plane_formats[0],
				                .components = {swizzle, swizzle, swizzle, swizzle},
				                .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
				        });
		}
		auto command_buffers = vk.device.allocateCommandBuffers({
		        .commandPool = *cmd_pool,
		        .commandBufferCount = num_slots,
		});

		for (auto [slot, cmd]: std::views::zip(slots, command_buffers))
		{
			slot.cmd = std::move(cmd);
			slot.fence = vk::raii::Fence(vk.device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
			slot.meta = make_buffer(encoder.meta_required_size(), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, false, std::format("pyrowave meta {}", stream_idx));
			slot.host_meta = make_buffer(encoder.meta_required_size(), vk::BufferUsageFlagBits::eTransferDst, true, std::format("pyrowave host meta {}", stream_idx));
			ensure_buffers(slot, target_size());
		}
	}

	~video_encoder_pyrowave()
	{
		std::vector<vk::Fence> fences;
		for (auto & slot: slots)
			fences.push_back(*slot.fence);
		if (vk.device.waitForFences(fences, true, 1'000'000'000) == vk::Result::eTimeout)
			U_LOG_E("pyrowave: timeout waiting for encoder %d", stream_idx);
	}

protected:
	void present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo compositor_sem, uint8_t slot_idx, uint64_t) override
	{
		auto & slot = slots[slot_idx];
		if (vk.device.waitForFences(*slot.fence, true, 1'000'000'000) == vk::Result::eTimeout)
		{
			U_LOG_E("pyrowave: timeout on stream %d", stream_idx);
			return;
		}

		if (auto new_bitrate = pending_bitrate.exchange(0))
			bitrate = new_bitrate;
		if (auto new_fps = pending_framerate.exchange(0))
			fps = new_fps;

		const auto target = target_size();
		ensure_buffers(slot, target);

		auto & cmd = slot.cmd;
		cmd.reset();
		cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

		encoder.encode(cmd,
		               input_views(cmd, y_cbcr),
		               {
		                       .meta = slot.meta,
		                       .bitstream = slot.bitstream,
		                       .bitstream_size = slot.bitstream.info().size,
		                       .target_size = target,
		               });

		cmd.copyBuffer(slot.meta, slot.host_meta, vk::BufferCopy{.size = slot.meta.info().size});
		cmd.copyBuffer(slot.bitstream, slot.host_bitstream, vk::BufferCopy{.size = slot.bitstream.info().size});
		pyrowave_core::memory_barrier(cmd,
		                         vk::PipelineStageFlagBits::eTransfer,
		                         vk::AccessFlagBits::eTransferWrite,
		                         vk::PipelineStageFlagBits::eHost,
		                         vk::AccessFlagBits::eHostRead);
		cmd.end();

		vk::CommandBufferSubmitInfo cmd_info{
		        .commandBuffer = *cmd,
		};
		compositor_sem.stageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;

		vk.device.resetFences(*slot.fence);
		std::unique_lock lock(vk.queue.mutex);
		vk.queue.queue.submit2(
		        vk::SubmitInfo2{
		                .waitSemaphoreInfoCount = 1,
		                .pWaitSemaphoreInfos = &compositor_sem,
		                .commandBufferInfoCount = 1,
		                .pCommandBufferInfos = &cmd_info,
		        },
		        *slot.fence);
	}

	std::optional<data> encode(uint8_t slot_idx, uint64_t) override
	{
		auto & slot = slots[slot_idx];
		if (vk.device.waitForFences(*slot.fence, true, 1'000'000'000) == vk::Result::eTimeout)
		{
			U_LOG_W("pyrowave: timeout on stream %d", stream_idx);
			return {};
		}

		vmaInvalidateAllocation(vk_allocator::instance(), slot.host_meta, 0, VK_WHOLE_SIZE);
		vmaInvalidateAllocation(vk_allocator::instance(), slot.host_bitstream, 0, VK_WHOLE_SIZE);

		bool ok = encoder.packetize(
		        slot.output,
		        std::span(slot.host_meta.data<const pyrowave_core::bitstream_packet>(), slot.host_meta.info().size / sizeof(pyrowave_core::bitstream_packet)),
		        std::span(slot.host_bitstream.data<const uint32_t>(), slot.host_bitstream.info().size / sizeof(uint32_t)));
		if (not ok)
			return {};

		return data{
		        .encoder = this,
		        .span = slot.output,
		};
	}
};
} // namespace wivrn
