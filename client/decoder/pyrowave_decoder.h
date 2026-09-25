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

#include "vk/pyrowave_common.h"

#include <array>
#include <atomic>
#include <memory>
#include <vector>

namespace wivrn
{
namespace pyrowave_core
{
class decoder : public wavelet_buffers
{
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

public:
	static void check_support(const device_caps & caps);
	decoder(vk::raii::Device & device, const device_caps & caps, const shader_map & shaders, int width, int height, chroma_subsampling chroma);
	void clear();
	bool push_packet(std::span<const uint8_t> data);

	bool decode_is_ready(bool allow_partial_frame, int pristine_bands = 2, float received_ratio = 0.9f) const;

	// Output planes must be in general layout and writable as storage images.
	// The previous decode must have completed.
	void decode(vk::raii::CommandBuffer & cmd, const view_buffers & views);

private:
	bool decode_packet(const bitstream_header & header, std::span<const uint8_t> packet);

	bool has_pristine_bands(int bands) const;

	void upload_payload();
	void dequant(vk::raii::CommandBuffer & cmd);
	void idwt(vk::raii::CommandBuffer & cmd, const view_buffers & views);
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

	static pyrowave_core::device_caps caps(vk::raii::PhysicalDevice & physical_device);
	image * get_free();

public:
	static bool supported();

	pyrowave_decoder(
	        vk::raii::Device & device,
	        vk::raii::PhysicalDevice & physical_device,
	        uint32_t vk_queue_family_index,
	        const wivrn::to_headset::video_stream_description & description,
	        uint8_t stream_index,
	        std::weak_ptr<scenes::stream> scene,
	        shard_accumulator * accumulator);
	~pyrowave_decoder();

	void push_data(std::span<std::span<const uint8_t>> data, uint64_t frame_index, bool partial) override;

	void frame_completed(
	        const from_headset::feedback & feedback,
	        const to_headset::video_stream_data_shard::view_info_t & view_info) override;

	vk::Sampler sampler() override
	{
		return *sampler_;
	}
};

} // namespace wivrn
