// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"

void kernel_main() {
    // Note: The argument index to get_compile_time_arg_val() must be a compile time constant.
    constexpr uint32_t Mt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);

    // We are going to read from these two circular buffers.
    // Note that indices have to be in sync with the reader kernel.
    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    // And write to this circular buffer.
    // Note that indices have to be in sync with the writer kernel.
    constexpr tt::CBIndex cb_out0 = tt::CBIndex::c_16;

    // FPU has a destination register, which is an array that can fit multiple tiles (details vary on data type).
    // For our case, FPU will add two tiles and produce a result that is a single tile.
    // We will instruct FPU to store the result in the destination register array at index 0.
    constexpr uint32_t dst_reg_idx = 0;

    // Initialize the Tensix Engine to perform an elementwise binary operation using circular buffers c_in0, c_in1 and c_out0.
    mm_init((uint32_t)cb_in0, (uint32_t)cb_in1, (uint32_t)cb_out0, 0);

    // Loop over all the tiles and perform the computation.
    // it's important to keep in mind that compute kernel runs on three different RISC-V processors.
    // One for unpacking, one for computing, and one for packing.
    // The compiler automatically compiles the same compute kernel code for all three processors,
    // relieving programmer from having to write different code for each core.
    for (uint32_t m = 0; m < Mt; m++) {
    for (uint32_t n = 0; n < Nt; n++) {

        // Initialize destination registers to zero for this output tile
        tile_regs_acquire();

        for (uint32_t k = 0; k < Kt; k++) {
            // Wait for next A and B tiles
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);

            // Accumulate A[m,k] * B[k,n] into dst register 0
            matmul_tiles(cb_in0, cb_in1, 0, 0, dst_reg_idx);

            // Consume input tiles
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }

        // Signal computation done for this output tile
        tile_regs_commit();

        // Pack and push result tile
        tile_regs_wait();
        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg_idx, cb_out0);
        cb_push_back(cb_out0, 1);
        tile_regs_release();
    }
}

}
