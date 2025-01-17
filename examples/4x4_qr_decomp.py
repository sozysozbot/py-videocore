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
    mov(rb0, r0)
    # STORE: rb0: [...a1, ...a2, ...a3, ...a4]
    

    """
    Compute u2 = a2 - (u1 * u1.dot(a2) / u1.dot(u1)).
    """

    mov(r1, 0.0)
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


    mov(r2, r0); fmul(r2, r2, r2); nop(); rotate(r3, r2, 14); fadd(r3, r3, r2)
    nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); 
    # r2 has [u1.dot(u1), ?, ?, ?, u2.dot(u2), ?, ?, ?, t3.dot(t3), ?, ?, ?, a4.dot(a4), ?, ?, ?]
    nop(); rotate(r2, r2, 12); mov(broadcast, r2) 
    # r5 has u2.dot(u2)

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1 has [...0vec, ...u2, ...0vec, ...0vec]

    mov(sfu_recip, r5)
    nop()
    nop()
    fmul(r1, r1, r4) 
    # r1 has [...0vec, ...(u2 / u2.dot(u2)), ...0vec, ...0vec]

    mov(ra1, r1)
    # STORE THIS FOR LATER USE
    # ra1: [...0vec, ...(u2 / u2.dot(u2)), ...0vec, ...0vec]

    nop(); rotate(r2, r0, 4)
    # r2: [...a4, ...u1, ...u2, ...t3]
    # r0: [...u1, ...u2, ...t3, ...a4]
    fmul(r2, r2, r0)
    # r2: [???, ???, u2*t3, ???]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); nop(); rotate(r2, r2, 8); mov(broadcast, r2)
    # r5 has u2.dot(t3)
    fmul(r1, r1, r5)
    # r1:  [...0vec, ...(u2 * u2.dot(t3) / u2.dot(u2)), ...0vec, ...0vec]
    nop(); rotate(r1, r1, 4); fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...u3, ...a4]
    
    """
    # Finish computing u3!!!!
    """

    """
    Next, compute u4.
    Note:
    ra0: [...(u1 / u1.dot(u1)), ...0vec, ...0vec, ...0vec]
    ra1: [...0vec, ...(u2 / u2.dot(u2)), ...0vec, ...0vec]
    """

    mov(r1, 0.0)
    ldi(null,[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    nop(); rotate(r2, r1, 12) # r2 has [...0vec, ...0vec, ...0vec, ...u1]
    fmul(r2, r2, r0) 
    # r2 has [...0vec, ...0vec, ...0vec, ...(u1*a4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); nop(); rotate(r2, r2, 4); mov(broadcast, r2)
    # r5 has u1.dot(a4)
    mov(r1, ra0); fmul(r1, r1, r5)
    # r1 has [...(u1 * u1.dot(a4) / u1.dot(u1)), ...0vec, ...0vec, ...0vec] 
    nop(); rotate(r1, r1, 12); fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...u3, ...t4] where t4 = a4 - (u1 * u1.dot(a4) / u1.dot(u1))

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    nop(); rotate(r2, r1, 8) # r2 has [...0vec, ...0vec, ...0vec, ...u2]
    fmul(r2, r2, r0) 
    # r2 has [...0vec, ...0vec, ...0vec, ...(u2*t4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); nop(); rotate(r2, r2, 4); mov(broadcast, r2)
    # r5 has u2.dot(t4)
    mov(r1, ra1); fmul(r1, r1, r5)
    # r1 has [...0vec, ...(u2 * u2.dot(t4) / u2.dot(u2)), ...0vec, ...0vec] 
    nop(); rotate(r1, r1, 8); fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...u3, ...s4] where s4 = t4 - (u2 * u2.dot(t4) / u2.dot(u2))
    
    mov(r2, r0); fmul(r2, r2, r2); nop(); rotate(r3, r2, 14); fadd(r3, r3, r2)
    nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); 
    # r2 has [u1.dot(u1), ?, ?, ?, u2.dot(u2), ?, ?, ?, u3.dot(u3), ?, ?, ?, s4.dot(s4), ?, ?, ?]
    nop(); rotate(r2, r2, 8); mov(broadcast, r2) 
    # r5 has u3.dot(u3)

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs')
    mov(sfu_recip, r5)
    nop()
    nop()
    fmul(ra2, r1, r4) 
    # STORE INTO ra2
    # ra2 has [...0vec, ...0vec, ...(u3 / u3.dot(u3)), ...0vec]

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    nop(); rotate(r2, r1, 4) # r2 has [...0vec, ...0vec, ...0vec, ...u3]
    fmul(r2, r2, r0) 
    # r2 has [...0vec, ...0vec, ...0vec, ...(u3*s4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); nop(); rotate(r2, r2, 4); mov(broadcast, r2)
    # r5 has u3.dot(s4)
    mov(r1, ra2); fmul(r1, r1, r5)
    # r1 has [...0vec, ...0vec, ...(u3 * u3.dot(s4) / u3.dot(u3)), ...0vec] 
    nop(); rotate(r1, r1, 4); fsub(r0, r0, r1)
    # r0: [...u1, ...u2, ...u3, ...u4] where u4 = t4 - (u3 * u3.dot(s4) / u3.dot(u3))

    mov(rb1, r0)
    """
    DONE computing U = rb1 = [...u1, ...u2, ...u3, ...u4]
    """

    """
    Now, let's compute Q
    """

    mov(r2, r0); fmul(r2, r2, r2); nop(); rotate(r3, r2, 14); fadd(r3, r3, r2)
    nop(); rotate(r2, r3, 15); fadd(r2, r3, r2); 
    # r2 has [u1.dot(u1), ?, ?, ?, u2.dot(u2), ?, ?, ?, u3.dot(u3), ?, ?, ?, u4.dot(u4), ?, ?, ?]
    mov(r1, 0.0)
    ldi(null,[0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],set_flags=True)
    mov(r1, r2, cond='zs') 
    # r1 has [u1.dot(u1), 0, 0, 0, u2.dot(u2), 0, 0, 0, u3.dot(u3), 0, 0, 0, u4.dot(u4), 0, 0, 0]
    nop(); rotate(r2, r1, 1); fadd(r1, r1, r2); nop(); rotate(r2, r1, 2); fadd(r1, r1, r2)
    # r1 has [u1.dot(u1) x 4, u2.dot(u2) x 4, u3.dot(u3) x 4, u4.dot(u4) x 4]
    mov(sfu_recipsqrt, r1)
    nop()
    nop()
    fmul(r0, r0, r4) 
    mov(rb2, r0)
    """
    DONE computing Q = rb2 = [...e1, ...e2, ...e3, ...e4]
    """

    """
    Okay, to finish off, we need R = [e1.dot(a1), 0, 0, 0, e1.dot(a2), e2.dot(a2), 0, 0, e1.dot(a3), e2.dot(a3), e3.dot(a3), 0 ... ]
    """

    mov(r1, 0.0)
    ldi(null,[0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1: [...e1, ...0vec, ...0vec, ...0vec]
    nop(); rotate(r2, r1, 4); fadd(r1, r1, r2); nop(); rotate(r2, r1, 8); fadd(r2, r1, r2)
    # r2: [...e1, ...e1, ...e1, ...e1]
    fmul(r2, r2, rb0)
    # r2: [...(e1*a1), ...(e1*a2), ...(e1*a3), ...(e1*a4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2)
    # r2: [e1.dot(a1), ?, ?, ?, e1.dot(a2), ?, ?, ?, e1.dot(a3), ?, ?, ?, e1.dot(a4), ?, ?, ?]

    mov(rb3, 0.0)
    ldi(null,[0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1],set_flags=True)
    mov(rb3, r2, cond='zs') 
    # rb3: [e1.dot(a1), 0, 0, 0, e1.dot(a2), 0, 0, 0, e1.dot(a3), 0, 0, 0, e1.dot(a4), 0, 0, 0]

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1: [...0vec, ...e2, ...0vec, ...0vec]
    nop(); rotate(r2, r1, 4); fadd(r2, r1, r2); rotate(r3, r1, 8); fadd(r2, r3, r2); 
    # r2: [...0vec, ...e2, ...e2, ...e2]
    fmul(r2, r2, rb0)
    # r2: [...0vec, ...(e2*a2), ...(e2*a3), ...(e2*a4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2)
    # r2: [0, ?, ?, ?, e2.dot(a2), ?, ?, ?, e2.dot(a3), ?, ?, ?, e2.dot(a4), ?, ?, ?]
    nop(); rotate(r2, r2, 1); 
    ldi(null,[1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1],set_flags=True)
    fadd(rb3, rb3, r2, cond='zs')

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1: [...0vec, ...0vec, ...e3, ...0vec]
    nop(); rotate(r2, r1, 4); fadd(r2, r1, r2)
    # r2: [...0vec, ...0vec, ...e3, ...e3]
    fmul(r2, r2, rb0)
    # r2: [...0vec, ...0vec, ...(e3*a3), ...(e3*a4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2)
    # r2: [0, ?, ?, ?, 0, ?, ?, ?, e2.dot(a3), ?, ?, ?, e2.dot(a4), ?, ?, ?]
    nop(); rotate(r2, r2, 2); 
    ldi(null,[1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1],set_flags=True)
    fadd(rb3, rb3, r2, cond='zs')

    mov(r1, 0.0)
    ldi(null,[1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],set_flags=True)
    mov(r1, r0, cond='zs') 
    # r1: [...0vec, ...0vec, ...0vec, ...e4]
    fmul(r2, r1, rb0)
    # r2: [...0vec, ...0vec, ...0vec, ...(e4*a4)]
    nop(); rotate(r3, r2, 14); fadd(r3, r3, r2); nop(); rotate(r2, r3, 15); fadd(r2, r3, r2)
    # r2: [0, ?, ?, ?, 0, ?, ?, ?, 0, ?, ?, ?, e4.dot(a4), ?, ?, ?]
    nop(); rotate(r2, r2, 3); 
    ldi(null,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],set_flags=True)
    fadd(rb3, rb3, r2, cond='zs')

    """
    R Complete!!!
    """

    mov(vpm, rb1)
    mov(vpm, rb2)
    mov(vpm, rb3)

    # Store the resulting vectors from VPM to the host memory (address=uniforms[1])
    setup_dma_store(nrows=3)
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
    out = drv.alloc(dim * dim * 3, 'float32')

    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(hello_world),
            uniforms=[inp.address, out.address]
            )

    a_mat = a_vec.reshape((dim, dim)).transpose()
    print(' A '.center(80, '='))
    print(a_mat)

    (u_t, q_t, r_t) = out.reshape((3, dim, dim))
    u = u_t.transpose()
    q = q_t.transpose()
    r = r_t.transpose()
    
    print(' U '.center(80, '='))
    print(u)
    print(' U^T @ U '.center(80, '='))
    np.set_printoptions(suppress=True)
    print(u_t @ u)
    np.set_printoptions(suppress=False)
   
    print(' Q '.center(80, '='))
    print(q)

    print(' Q^T @ Q '.center(80, '='))
    np.set_printoptions(suppress=True)
    print(q_t @ q)
    np.set_printoptions(suppress=False)

    print(' R '.center(80, '='))
    print(r)
    print(' A - Q@R '.center(80, '='))
    np.set_printoptions(suppress=True)
    print(np.abs(a_mat-(q@r)))
    np.set_printoptions(suppress=False)

