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

// Port of pyrowave_common.cpp from Granite to vulkan-hpp.
// See https://github.com/Themaister/pyrowave
//
// Copyright (c) 2025 Hans-Kristian Arntzen
// SPDX-License-Identifier: MIT

#include "pyrowave_common.h"

#include <cstring>

namespace wivrn::pyrowave_core
{

device_caps device_caps::query(vk::raii::PhysicalDevice & physical_device, uint32_t api_version, std::span<const char * const> enabled_extensions)
{
	auto has_ext = [&](const char * name) {
		return std::ranges::any_of(enabled_extensions, [&](const char * e) { return strcmp(e, name) == 0; });
	};
	const bool has_vk12 = api_version >= VK_API_VERSION_1_2;

	auto [props2, subgroup_props] = physical_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
	auto [feat2, feat_16bit] = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDevice16BitStorageFeatures>();

	device_caps caps{
	        .subgroup_ops = subgroup_props.supportedOperations,
	        .subgroup_size = subgroup_props.subgroupSize,
	        .min_subgroup_size = subgroup_props.subgroupSize,
	        .max_subgroup_size = subgroup_props.subgroupSize,
	        .push_descriptor = has_ext(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME),
	        .shader_int16 = bool(feat2.features.shaderInt16),
	        .storage_buffer_16bit = (has_vk12 or has_ext(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) and feat_16bit.storageBuffer16BitAccess,
	        .storage_image_write_without_format = bool(feat2.features.shaderStorageImageWriteWithoutFormat),
	        .vendor_id = props2.properties.vendorID,
	        .max_texel_buffer_elements = props2.properties.limits.maxTexelBufferElements,
	};

	if (has_vk12 or has_ext(VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
	{
		auto feat = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDevice8BitStorageFeatures>();
		caps.storage_buffer_8bit = feat.get<vk::PhysicalDevice8BitStorageFeatures>().storageBuffer8BitAccess;
	}

	if (has_vk12 or has_ext(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
	{
		auto feat = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceShaderFloat16Int8Features>();
		caps.shader_float16 = feat.get<vk::PhysicalDeviceShaderFloat16Int8Features>().shaderFloat16;
	}

	if (has_vk12 or has_ext(VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME))
	{
		auto props = physical_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
		caps.driver_id = props.get<vk::PhysicalDeviceDriverProperties>().driverID;
	}

	if (api_version >= VK_API_VERSION_1_3 or has_ext(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
	{
		auto feat = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceSubgroupSizeControlFeatures>();
		auto props = physical_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupSizeControlProperties>();
		const auto & size_control = feat.get<vk::PhysicalDeviceSubgroupSizeControlFeatures>();
		const auto & size_control_props = props.get<vk::PhysicalDeviceSubgroupSizeControlProperties>();
		caps.subgroup_size_control = size_control.subgroupSizeControl;
		caps.compute_full_subgroups = size_control.computeFullSubgroups;
		if (caps.subgroup_size_control)
		{
			caps.min_subgroup_size = size_control_props.minSubgroupSize;
			caps.max_subgroup_size = size_control_props.maxSubgroupSize;
			caps.required_subgroup_size_compute = bool(size_control_props.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute);
		}
	}
	return caps;
}

std::optional<device_caps::subgroup_config> device_caps::subgroup_for(int min_log2, int max_log2, uint32_t local_size_x) const
{
	const uint32_t lo = 1u << min_log2;
	const uint32_t hi = 1u << max_log2;

	if (not subgroup_size_control)
	{
		if (subgroup_size < lo or subgroup_size > hi)
			return std::nullopt;
		return subgroup_config{};
	}

	if (not compute_full_subgroups)
		return std::nullopt;

	// Any size the device may use is fine, let the driver choose
	if (lo <= min_subgroup_size and hi >= max_subgroup_size)
		return subgroup_config{
		        // Only allowed when the workgroup size is a multiple of the largest size
		        .full_subgroups = local_size_x % max_subgroup_size == 0,
		        .varying = true,
		};

	if (not required_subgroup_size_compute)
		return std::nullopt;

	// Prefer the smallest subgroup size
	uint32_t size = std::max(lo, min_subgroup_size);
	if (size > std::min(hi, max_subgroup_size))
		return std::nullopt;
	return subgroup_config{
	        .required_size = size,
	        .full_subgroups = local_size_x % size == 0,
	};
}

compute_pipeline::compute_pipeline(
        vk::raii::Device & device,
        const shader_map & shaders,
        const char * name,
        std::initializer_list<vk::DescriptorType> bindings,
        uint32_t push_constant_size,
        const device_caps::subgroup_config & subgroup,
        const vk::SpecializationInfo * specialization)
{
	std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
	for (auto type: bindings)
		layout_bindings.push_back({
		        .binding = uint32_t(layout_bindings.size()),
		        .descriptorType = type,
		        .descriptorCount = 1,
		        .stageFlags = vk::ShaderStageFlagBits::eCompute,
		});

	set_layout = vk::raii::DescriptorSetLayout(
	        device,
	        vk::DescriptorSetLayoutCreateInfo{
	                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
	                .bindingCount = uint32_t(layout_bindings.size()),
	                .pBindings = layout_bindings.data(),
	        });

	vk::PushConstantRange push_range{
	        .stageFlags = vk::ShaderStageFlagBits::eCompute,
	        .size = push_constant_size,
	};
	layout = vk::raii::PipelineLayout(
	        device,
	        vk::PipelineLayoutCreateInfo{
	                .setLayoutCount = 1,
	                .pSetLayouts = &*set_layout,
	                .pushConstantRangeCount = push_constant_size ? 1u : 0u,
	                .pPushConstantRanges = &push_range,
	        });

	const auto & spirv = shaders.at(name);
	vk::raii::ShaderModule module(
	        device,
	        vk::ShaderModuleCreateInfo{
	                .codeSize = spirv.size() * sizeof(uint32_t),
	                .pCode = spirv.data(),
	        });

	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo required_size{
	        .requiredSubgroupSize = subgroup.required_size,
	};
	vk::PipelineShaderStageCreateFlags flags;
	if (subgroup.full_subgroups)
		flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroups;
	if (subgroup.varying)
		flags |= vk::PipelineShaderStageCreateFlagBits::eAllowVaryingSubgroupSize;

	pipeline = vk::raii::Pipeline(
	        device,
	        nullptr,
	        vk::ComputePipelineCreateInfo{
	                .stage = {
	                        .pNext = subgroup.required_size ? &required_size : nullptr,
	                        .flags = flags,
	                        .stage = vk::ShaderStageFlagBits::eCompute,
	                        .module = *module,
	                        .pName = "main",
	                        .pSpecializationInfo = specialization,
	                },
	                .layout = *layout,
	        });
}

void compute_pipeline::push_descriptors(vk::raii::CommandBuffer & cmd, std::initializer_list<descriptor> descriptors) const
{
	pyrowave_core::push_descriptors(cmd, vk::PipelineBindPoint::eCompute, *layout, descriptors);
}

void push_descriptors(vk::raii::CommandBuffer & cmd, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, std::initializer_list<descriptor> descriptors)
{
	std::array<vk::WriteDescriptorSet, 8> writes;
	assert(descriptors.size() <= writes.size());
	uint32_t i = 0;
	for (const auto & d: descriptors)
	{
		writes[i] = vk::WriteDescriptorSet{
		        .dstBinding = i,
		        .descriptorCount = 1,
		        .descriptorType = d.type,
		        .pImageInfo = &d.image,
		        .pBufferInfo = &d.buffer,
		        .pTexelBufferView = &d.texel_buffer,
		};
		++i;
	}
	cmd.pushDescriptorSetKHR(bind_point, layout, 0, vk::ArrayProxy<const vk::WriteDescriptorSet>(i, writes.data()));
}

void memory_barrier(vk::raii::CommandBuffer & cmd, vk::PipelineStageFlags src_stage, vk::AccessFlags src_access, vk::PipelineStageFlags dst_stage, vk::AccessFlags dst_access)
{
	cmd.pipelineBarrier(src_stage, dst_stage, {}, vk::MemoryBarrier{.srcAccessMask = src_access, .dstAccessMask = dst_access}, {}, {});
}

void discard_to_general(vk::raii::CommandBuffer & cmd, vk::Image image, vk::PipelineStageFlags src_stage, vk::PipelineStageFlags dst_stage, vk::AccessFlags dst_access)
{
	cmd.pipelineBarrier(
	        src_stage,
	        dst_stage,
	        {},
	        {},
	        {},
	        vk::ImageMemoryBarrier{
	                .dstAccessMask = dst_access,
	                .oldLayout = vk::ImageLayout::eUndefined,
	                .newLayout = vk::ImageLayout::eGeneral,
	                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
	                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
	                .image = image,
	                .subresourceRange = {
	                        .aspectMask = vk::ImageAspectFlagBits::eColor,
	                        .levelCount = vk::RemainingMipLevels,
	                        .layerCount = vk::RemainingArrayLayers,
	                },
	        });
}

wavelet_buffers::wavelet_buffers(vk::raii::Device & device, int width, int height, chroma_subsampling chroma) :
        device(&device),
        width(width),
        height(height),
        chroma(chroma)
{
	aligned_width = std::max(align(width, alignment), minimum_image_size);
	aligned_height = std::max(align(height, alignment), minimum_image_size);

	init_samplers();
	allocate_images();
	init_block_meta();
}

void wavelet_buffers::transition_wavelet_images(vk::raii::CommandBuffer & cmd)
{
	discard_to_general(cmd, wavelet_img_high_res, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
	discard_to_general(cmd, wavelet_img_low_res, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite);
}

void wavelet_buffers::init_samplers()
{
	vk::SamplerCreateInfo info{
	        .magFilter = vk::Filter::eNearest,
	        .minFilter = vk::Filter::eNearest,
	        .mipmapMode = vk::SamplerMipmapMode::eNearest,
	        .addressModeU = vk::SamplerAddressMode::eMirroredRepeat,
	        .addressModeV = vk::SamplerAddressMode::eMirroredRepeat,
	        .addressModeW = vk::SamplerAddressMode::eMirroredRepeat,
	        .maxLod = VK_LOD_CLAMP_NONE,
	};
	mirror_repeat_sampler = vk::raii::Sampler(*device, info);

	info.addressModeU = vk::SamplerAddressMode::eClampToBorder;
	info.addressModeV = vk::SamplerAddressMode::eClampToBorder;
	info.addressModeW = vk::SamplerAddressMode::eClampToBorder;
	info.borderColor = vk::BorderColor::eFloatTransparentBlack;
	border_sampler = vk::raii::Sampler(*device, info);
}

void wavelet_buffers::allocate_images()
{
	vk::ImageCreateInfo info{
	        .imageType = vk::ImageType::e2D,
	        .format = vk::Format::eR16Sfloat,
	        .extent = {uint32_t(aligned_width / 2), uint32_t(aligned_height / 2), 1},
	        .mipLevels = wavelet_fp16_levels,
	        .arrayLayers = num_frequency_bands_per_level * num_components,
	        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
	};
	wavelet_img_high_res = image_allocation(*device, info, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave wavelet high res");

	// For the lowest level bands, we want to maintain precision as much as possible and bandwidth here is trivial.
	info.format = vk::Format::eR32Sfloat;
	info.mipLevels = decomposition_levels - wavelet_fp16_levels;
	info.extent.width >>= wavelet_fp16_levels;
	info.extent.height >>= wavelet_fp16_levels;
	wavelet_img_low_res = image_allocation(*device, info, {.usage = VMA_MEMORY_USAGE_AUTO}, "pyrowave wavelet low res");

	for (int component = 0; component < num_components; component++)
	{
		for (int level = 0; level < decomposition_levels; level++)
		{
			bool high_res = level < wavelet_fp16_levels;
			vk::ImageViewCreateInfo view_info{
			        .image = high_res ? vk::Image(wavelet_img_high_res) : vk::Image(wavelet_img_low_res),
			        .viewType = vk::ImageViewType::e2DArray,
			        .format = high_res ? vk::Format::eR16Sfloat : vk::Format::eR32Sfloat,
			        .subresourceRange = {
			                .aspectMask = vk::ImageAspectFlagBits::eColor,
			                .baseMipLevel = uint32_t(high_res ? level : level - wavelet_fp16_levels),
			                .levelCount = 1,
			                .baseArrayLayer = uint32_t(4 * component),
			                .layerCount = 4,
			        },
			};
			component_layer_views.emplace_back(*device, view_info);

			view_info.viewType = vk::ImageViewType::e2D;
			view_info.subresourceRange.layerCount = 1;
			component_ll_views.emplace_back(*device, view_info);
		}
	}
}

void wavelet_buffers::accumulate_block_mapping(int blocks_x_8x8, int blocks_y_8x8)
{
	int blocks_x_32x32 = (blocks_x_8x8 + 3) / 4;
	int blocks_y_32x32 = (blocks_y_8x8 + 3) / 4;

	for (int y = 0; y < blocks_y_32x32; y++)
	{
		for (int x = 0; x < blocks_x_32x32; x++)
		{
			block_32x32_to_8x8_mapping.push_back({
			        .block_offset_8x8 = block_count_8x8 + 4 * y * blocks_x_8x8 + 4 * x,
			        .block_stride_8x8 = blocks_x_8x8,
			        .block_width_8x8 = std::min<int>(4, blocks_x_8x8 - 4 * x),
			        .block_height_8x8 = std::min<int>(4, blocks_y_8x8 - 4 * y),
			});
			block_count_32x32++;
		}
	}

	block_count_8x8 += blocks_x_8x8 * blocks_y_8x8;
}

void wavelet_buffers::init_block_meta()
{
	for (int level = decomposition_levels - 1; level >= 0; level--)
	{
		for (int component = 0; component < num_components; component++)
		{
			if (skip(level, component, chroma))
				continue;

			for (int band = first_band(level); band < 4; band++)
			{
				int blocks_x_8x8 = (level_width(level) + 7) / 8;
				int blocks_y_8x8 = (level_height(level) + 7) / 8;
				int blocks_x_32x32 = (level_width(level) + 31) / 32;
				int blocks_y_32x32 = (level_height(level) + 31) / 32;

				block_meta[component][level][band] = {
				        block_count_8x8,
				        blocks_x_8x8,
				        block_count_32x32,
				        blocks_x_32x32,
				        blocks_x_32x32 * blocks_y_32x32,
				};

				accumulate_block_mapping(blocks_x_8x8, blocks_y_8x8);
			}
		}
	}
}

} // namespace wivrn::pyrowave_core
