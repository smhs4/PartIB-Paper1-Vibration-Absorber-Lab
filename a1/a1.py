#! /usr/bin/env python3

import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

def MLKF_building(m, k, l, f, n_tmds=0):
    """Return mass, damping, and stiffness matrices for a building with the possibility of n TMDs"""

    n_floors = len(m)

    # Create the matrices with room for the TMDs
    M = np.zeros((n_floors + n_tmds, n_floors + n_tmds))
    L = np.zeros_like(M)
    K = np.zeros_like(M)
    F = np.concatenate((np.array(f), np.zeros(n_tmds)))
    

    # Populate the diagonal elements for the building
    for i in range(n_floors):
        M[i, i] = m[i]
        
        # Populate K
        if i < n_floors-1 :
            K[i,i] += k[i+1]
            K[i+1,i+1] += k[i+1]
            K[i+1,i] -= k[i+1]
            K[i,i+1] -= k[i+1]

        if i == 0:
            K[i,i] += k[i]

    
        # Populate L
        if i < n_floors-1 :
            L[i,i] += l[i+1]
            L[i+1,i+1] += l[i+1]
            L[i+1,i] -= l[i+1]
            L[i,i+1] -= l[i+1]

        if i == 0:
            L[i,i] += l[i]

    return M, L, K, F

def add_TMD(M, L, K, m_tmd, k_tmd, l_tmd, floor_num, tmd_idx, m):

    print("recieved:" , L)
    """Add the effect of a TMD to the building matrices at the specified index"""
    idx = tmd_idx + len(m)
    
    # Set the mass for the TMD
    M[idx,idx] = m_tmd

    # Set the lambda for the TMD
    L[idx,idx] += l_tmd
    print( "1:", L)
    L[floor_num-1,floor_num-1] += l_tmd
    print( "2:", L)
    L[floor_num-1, idx] -= l_tmd
    print("3:", L)
    L[idx,floor_num-1] -= l_tmd
    print("4:", L)
    
    # Set the K for the TMD
    K[idx,idx] += k_tmd
    K[floor_num-1,floor_num-1] += k_tmd
    K[floor_num-1, idx] -= k_tmd
    K[idx,floor_num-1] -= k_tmd


    return M, L, K


def build_matrix(m, l, k, f, m_tmds, l_tmds, k_tmds, floor_tmds):
    n_tmds = len(m_tmds)
    M, L, K, F = MLKF_building(m, k, l, f, n_tmds)

    if n_tmds > 0:
        for i in range(len(m_tmds)) :
            M, L, K = add_TMD(M, L, K, m_tmds[i], k_tmds[i], l_tmds[i], floor_tmds[i], i, m)
            print("here:" , L)

    return M, L, K, F

def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )


def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    mm = M.diagonal()

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = (F - L@xv[1] - K@xv[0]) / mm
        s = np.concatenate((xv[1], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(len(mm) * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[0:len(mm), :].T


def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(fig, hz, sec, M, L, K, F, show_phase=None):

    """Plot frequency and time domain responses"""

    # Generate response data

    f_response = freq_response(hz * 2*np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # Determine suitable legends

    f_legends = (
        'm{} peak {:.4g} metre at {:.4g} Hz'.format(
            i+1,
            f_amplitude[m][i],
            hz[m]
        )
        for i, m in enumerate(np.argmax(f_amplitude, axis=0))
    )

    equilib = np.abs(freq_response([0], M, L, K, F))[0]         # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec)-1)

    t_legends = (
        'm{} settled to 2% beyond {:.4g} sec'.format(
            i+1,
            sec[lastbig[i]]
        )
        for i, _ in enumerate(t_response.T)
    )

    # Create plot

    fig.clear()

    if show_phase is not None:
        ax = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3)
        ]
        ax[1].sharex(ax[0])
    else:
        ax = [
            fig.add_subplot(2, 1, 1),
            fig.add_subplot(2, 1, 2)
        ]

    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    ax[0].legend(ax[0].plot(hz, f_amplitude), f_legends)

    if show_phase is not None:
        p_legends = (f'm{i+1}' for i in range(f_response.shape[1]))

        f_phases = f_response
        if show_phase == 0:
            ax[1].set_title(f'Phase of frequency domain response to sinusoidal force')
        else:
            f_phases /= f_response[:, show_phase-1:show_phase]
            ax[1].set_title(f'Phase, relative to m{show_phase}, of frequency domain response to sinusoidal force')
        f_phases = np.degrees(np.angle(f_phases))

        ax[1].set_xlabel('Frequency/hertz')
        ax[1].set_ylabel('Phase/°')
        ax[1].legend(ax[1].plot(hz, f_phases), p_legends)

    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/metre')
    ax[-1].legend(ax[-1].plot(sec, t_response), t_legends)

    fig.tight_layout()

def main():

    """Main program"""

    # Enter values of m, k and lamda for each floor, make sure the m, k, l(ambda) and f arrays all have the same length with the each index 
    # correspoinding to the values of each floor.

    m = [1.83,1.83,1.83]
    k = [4200,4200,4200]
    l = [1, 1, 1 ]
    f = [0.5, 0, 0]

    # Here you can add the value of the tuned mass dampners that you want. again make sure the four arrays have the same length.
    m_tmds = [0.15]
    l_tmds = [1]
    k_tmds = [80]
    floor_tmds = [1] # This allows you to choose which floor you want to place the tuned mass damper on

    M, L, K, F = build_matrix(m, l, k, f, m_tmds, l_tmds, k_tmds, floor_tmds)
    print("M:",M)
    print("L:", L)
    print("K:", K)
    
    # Generate frequency and time arrays

    hz = np.linspace(0, 15, 10001)
    sec = np.linspace(0, 30, 10001)

    # Plot results

    fig = plt.figure()
    plot(fig, hz, sec, M, L, K, F)
    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    plt.show()


if __name__ == '__main__':
    main()
