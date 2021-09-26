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

    mov(r0, vpm)
    # r0: [...a1, ...a2, ...a3, ...a4]
    mov(r1, 0.0)
    # r1: [...0vec, ...0vec, ...0vec, ...0vec]

    """
    Compute u2 = a2 - (u1 * u1.dot(a2) / u1.dot(u1)).
    """

    ldi(null,[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1: [...u1, ...0vec, ...0vec, ...0vec]

    mov(r2, r1) 
    # r2: [...u1, ...0vec, ...0vec, ...0vec]

    # r2: u1=[A, B, C, D, ...]
    fmul(r2, r2, r2) 
    # r2: [A^2, B^2, C^2, D^2, ...]
    nop(); rotate(r3, r2, 14)
    # r2: [A^2, B^2, C^2, D^2, ...], r3: [C^2, D^2, 0, 0, ...]
    fadd(r3, r3, r2)
    # r3: [A^2 + C^2, B^2 + D^2, C^2, D^2, ...]
    nop(); rotate(r2, r3, 15)
    # r3: [A^2 + C^2, B^2 + D^2, ...], r2: : [B^2 + D^2, C^2, ...]
    fadd(r2, r3, r2); 
    # r2: [A^2+B^2+C^2+D^2, ...]
    mov(broadcast, r2) 
    # r5 has u1.dot(u1)

    
    nop(); rotate(r2, r1, 4) # r2 has [...0vec, ...u1, ...0vec, ...0vec]

    mov(sfu_recip, r5)
    nop()
    nop()
    fmul(r1, r1, r4) 
    # r1 has [...(u1 / u1.dot(u1)), ...0vec, ...0vec, ...0vec]

    mov(ra0, r1)
    # STORE THIS FOR LATER USE
    # ra0: [...(u1 / u1.dot(u1)), ...0vec, ...0vec, ...0vec]

    fmul(r2, r2, r0) 
    # r2 has [...0vec, ...(u1*a2), ...0vec, ...0vec]
    # let r2 be [...0vec, A, B, C, D, ...0vec, ...0vec]
    nop(); rotate(r3, r2, 14)
    # r2: [0, 0, 0, 0, A, B, C, D, ...0vec, ...0vec]
    # r3: [0, 0, A, B, C, D, 0, 0, ...0vec, ...0vec]
    fadd(r3, r3, r2)
    # r3: [0, 0, A, B, A+C, B+D, 0, 0, ...0vec, ...0vec]
    nop(); rotate(r2, r3, 15); 
    # r3: [0, 0, A, B, A+C, B+D, 0, 0, ...0vec, ...0vec]
    # r2: [0, A, B, A+C, B+D, 0, 0, 0, ...0vec, ...0vec]
    fadd(r2, r3, r2); 
    # r2: [...?, u1.dot(a2), ...?, ...?]
    nop(); rotate(r2, r2, 12); mov(broadcast, r2)
    # r5 has u1.dot(a2)
    
    nop(); rotate(r1, r1, 4); fmul(r1, r1, r5)
    # r1 has [...0vec, ...(u1 * u1.dot(a2) / u1.dot(u1)), ...0vec, ...0vec]

    fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...a3, ...a4]
    """
    Finish computing u2!!!!
    """

    """
    Next, compute u3.
    Note that ra0: [...(u1 / u1.dot(u1)), ...0vec, ...0vec, ...0vec] can be used
    """

    ldi(null,[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    nop(); rotate(r2, r1, 8) # r2 has [...0vec, ...0vec, ...u1, ...0vec]
    fmul(r2, r2, r0) 
    # r2 has [...0vec, ...0vec, ...(u1*a3), ...0vec]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); nop(); rotate(r2, r2, 8); mov(broadcast, r2)
    # r5 has u1.dot(a3)
    mov(r1, ra0); fmul(r1, r1, r5)
    # r1 has [...(u1 * u1.dot(a3) / u1.dot(u1)), ...0vec, ...0vec, ...0vec] 
    nop(); rotate(r1, r1, 8); fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...t3, ...a4] where t3 = a3 - (u1 * u1.dot(a3) / u1.dot(u1))
    # Projecting t3 further onto u2 is fine. Hence we want u3 = t3 - (u2 * u2.dot(t3) / u2.dot(u2)) 

    mov(vpm, r0)
    mov(vpm, 0.0)

    # Store the resulting vectors from VPM to the host memory (address=uniforms[1])
    setup_dma_store(nrows=2)
    start_dma_store(uniform)
    wait_dma_store()

    # Finish the thread
    exit()

with Driver() as drv:
    dim = 4
    # Input vector
    a_vec = np.random.random(dim * dim).astype('float32')

    # Copy vectors to shared memory for DMA transfer
    inp = drv.copy(np.r_[a_vec])
    out = drv.alloc(dim * dim * 2, 'float32')

    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(hello_world),
            uniforms=[inp.address, out.address]
            )

    a_mat = a_vec.reshape((dim, dim)).transpose()
    print(' A '.center(80, '='))
    print(a_mat)

    (q_t, r_t) = out.reshape((2, dim, dim))
    q = q_t.transpose()
    r = r_t.transpose()

    print(' Q '.center(80, '='))
    print(q)
    print(' R '.center(80, '='))
    print(r)
    print(' Q^T @ Q '.center(80, '='))
    np.set_printoptions(suppress=True)
    print(q_t @ q)
    np.set_printoptions(suppress=False)
    print(' error '.center(80, '='))
    print(np.abs(a_mat-(q@r)))

