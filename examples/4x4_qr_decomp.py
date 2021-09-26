import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def hello_world(asm):
    # Load a single vector of length 16 from the host memory (address=uniforms[0]) to VPM
    setup_dma_load(nrows=1)
    start_dma_load(uniform)
    wait_dma_load()

    # Setup VPM read/write operations
    setup_vpm_read(nrows=1)
    setup_vpm_write()

    # Move a
    mov(r0, vpm)
    mov(vpm, r0)
    mov(vpm, r0)

    # Store the resulting vectors from VPM to the host memory (address=uniforms[1])
    setup_dma_store(nrows=2)
    start_dma_store(uniform)
    wait_dma_store()

    # Finish the thread
    exit()

with Driver() as drv:
    dim = 4
    # Input vectors
    a = np.random.random(dim * dim).astype('float32')
    b = np.random.random(dim * dim).astype('float32')

    # Copy vectors to shared memory for DMA transfer
    inp = drv.copy(np.r_[a])
    out = drv.alloc(dim * dim * 2, 'float32')

    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(hello_world),
            uniforms=[inp.address, out.address]
            )

    a_mat = a.reshape((dim, dim))
    print(' A '.center(80, '='))
    print(a_mat)
    print(' Q '.center(80, '='))
    (q, r) = out.reshape((2, dim, dim))
    print(q)
    print(' R '.center(80, '='))
    print(r)
    print(' error '.center(80, '='))
    print(np.abs(a_mat-(q@r)))

