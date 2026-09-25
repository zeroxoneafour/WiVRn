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

#include "video_encoder.h"
#include "vk/pyrowave_common.h"

#include <array>
#include <vector>

namespace wivrn
{
namespace pyrowave_core
{
class encoder : public wavelet_buffers
{
	compute_pipeline dwt_pipeline[2]; // without and with DC shift
	compute_pipeline quant_pipeline;
	compute_pipeline analyze_pipeline;
	compute_pipeline analyze_finalize_pipeline;
	compute_pipeline resolve_pipeline;
	compute_pipeline packing_pipeline;

	buffer_allocation bucket_buffer, meta_buffer, block_stat_buffer, payload_data, quant_buffer;

	uint32_t sequence_count = 0;

	buffer_allocation make_buffer(vk::DeviceSize size, const char * name);

public:
	struct bitstream_buffers
	{
		vk::Buffer meta;
		vk::Buffer bitstream;
		vk::DeviceSize bitstream_size;
		size_t target_size;
	};

	static void check_support(const device_caps & caps);
	encoder(vk::raii::Device & device, const device_caps & caps, const shader_map & shaders, int width, int height, chroma_subsampling chroma);

	vk::DeviceSize meta_required_size() const
	{
		return block_count_32x32 * sizeof(bitstream_packet);
	}

	// Records the encode commands, input planes must be in general layout
	void encode(vk::raii::CommandBuffer & cmd, const view_buffers & views, const bitstream_buffers & buffers);

	// Build the frame from the block packets written by encode, returns false on error
	bool packetize(std::vector<uint8_t> & out, std::span<const bitstream_packet> meta, std::span<const uint32_t> bitstream) const;

private:
	float get_quant_rdo_distortion_scale(int level, int component, int band) const;

	float get_quant_resolution(int level, int component, int band) const;

	float get_noise_power_normalized_quant_resolution(int level, int component, int band) const;

	void dwt(vk::raii::CommandBuffer & cmd, const view_buffers & views);
	void quant(vk::raii::CommandBuffer & cmd);
	void analyze_rdo(vk::raii::CommandBuffer & cmd);
	void resolve_rdo(vk::raii::CommandBuffer & cmd, size_t target_payload_size);
	void block_packing(vk::raii::CommandBuffer & cmd, const bitstream_buffers & buffers);
};
} // namespace pyrowave_core

class video_encoder_pyrowave : public video_encoder
{
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

	buffer_allocation make_buffer(vk::DeviceSize size, vk::BufferUsageFlags usage, bool host, const std::string & name);
	pyrowave_core::view_buffers input_views(vk::raii::CommandBuffer & cmd, vk::Image image);

	size_t target_size() const;

	void ensure_buffers(slot_t & slot, size_t target);

public:
	static bool supported(vk_bundle & vk);
	video_encoder_pyrowave(vk_bundle & vk, const encoder_settings & settings, uint8_t stream_idx);
	~video_encoder_pyrowave();

protected:
	void present_image(vk::Image y_cbcr, vk::SemaphoreSubmitInfo compositor_sem, uint8_t slot_idx, uint64_t) override;

	std::optional<data> encode(uint8_t slot_idx, uint64_t) override;
};
} // namespace wivrn
