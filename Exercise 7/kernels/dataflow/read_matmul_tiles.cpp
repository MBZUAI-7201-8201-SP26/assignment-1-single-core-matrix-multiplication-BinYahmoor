// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Read parameters from the kernel's runtime arguments.
    int arg_idx = 0;
    uint32_t in0_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t in1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Mt = get_arg_val<uint32_t>(arg_idx++);
uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);
uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);

    // The circular buffers to read the tiles into
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;

    // Get the tile size used in the circular buffers. We assume the
    // circular buffers are created with the same tile size as the DRAM
    // buffers (which is most often the case).
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // Create address generators for the input buffers. Address generators can determine
    // physical address based on the provided data layout and base address.
    // Start by extracting the tensor layout parameters from the compile-time arguments.
    // Recall that compile-time arguments are stored as a vector of uint32_t values.
    // TensorAccessorArgs is a clean way to extract the appropriate number of
    // these uint32_t values and store them in an object.
    constexpr auto in0_layout_args = TensorAccessorArgs<0>();
    // Then, construct the address generator for the input buffer.
    // Observe that here we are just constructing the address generator object, but not using it yet.
    // It will be used in the loop below to determine the address to read the tiles from.
    const auto in0_addr_gen = TensorAccessor(in0_layout_args, in0_base_addr, tile_size_bytes);
    // Repeat for the second input buffer. next_compile_time_args_offset() is a clean way to
    // determine the index of the next compile-time argument (after TensorAccessorArgs read
    // the number of arguments it required above) without having to hard-code the index.
    constexpr auto in1_layout_args = TensorAccessorArgs<in0_layout_args.next_compile_time_args_offset()>();
    // Finally, construct the address generator for the second input buffer.
    const auto in1_addr_gen = TensorAccessor(in1_layout_args, in1_base_addr, tile_size_bytes);

    // Loop over all the tiles and read them into the circular buffers.
    for (uint32_t m = 0; m < Mt; m++) {
    for (uint32_t n = 0; n < Nt; n++) {
        for (uint32_t k = 0; k < Kt; k++) {

            cb_reserve_back(cb_in0, 1);
            cb_reserve_back(cb_in1, 1);

            uint32_t cb_in0_addr = get_write_ptr(cb_in0);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);

            uint32_t a_tile_idx = m * Kt + k;
            uint32_t b_tile_idx = k * Nt + n;

            noc_async_read_tile(a_tile_idx, in0_addr_gen, cb_in0_addr);
            noc_async_read_tile(b_tile_idx, in1_addr_gen, cb_in1_addr);

            noc_async_read_barrier();

            cb_push_back(cb_in0, 1);
            cb_push_back(cb_in1, 1);
        }
    }
}

}
