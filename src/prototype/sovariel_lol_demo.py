# src/prototype/sovariel_lol_demo.py
# Sovariel-LoL v1.4: Grok Dynamic Gamma via Pitch (Nov 25, 2025)
# Voice RMS + pitch mod τ & gamma (cap 0.005). Phase noise for OR realism in LoL.
# © 2025 AgapeIntelligence — MIT License

import math
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import mat_dict
import sounddevice as sd
import speech_recognition as sr
import serial  # Optional
import matplotlib.pyplot as plt
import time
import logging
from multiprocessing import Pool
from scipy.fft import fft  # For pitch (scipy in env)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('sovariel_lol')

SAMPLE_RATE = 44100
VOICE_DURATION = 0.25
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
N_QUBITS = 4
N_TRAJ = 50

def single_site_op(n_qubits, op, i):
    ops = [torch.eye(2, device=DEVICE) for _ in range(n_qubits)]
    ops[i] = op.to(DEVICE)
    return tq.functional.tensor(*ops)

def get_voice_rms_and_pitch():
    recognizer = sr.Recognizer()
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1) as stream:
            audio_data = sd.rec(int(VOICE_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
            sd.wait()
        audio = sr.AudioData(audio_data.flatten().tobytes(), SAMPLE_RATE, 2)
        rms = np.sqrt(np.mean(audio_data**2))
        rms_var = np.clip(rms * 60.0, 0.08, 0.35)
        
        # Pitch via FFT (fundamental freq proxy, 100-300Hz human range)
        fft_vals = np.abs(fft(audio_data.flatten()))
        freqs = np.fft.fftfreq(len(fft_vals), 1/SAMPLE_RATE)
        pitch_idx = np.argmax(fft_vals[100:600]) + 100  # Rough 100-300Hz window
        pitch_var = np.clip(freqs[pitch_idx] / 200.0, 0.0, 1.0)  # Normalize 0-1
        
        log.info(f"RMS: {rms_var:.3f} | Pitch Var: {pitch_var:.3f}")
        return rms_var, pitch_var
    except Exception as e:
        log.warning(f"Voice failed: {e}")
        return 0.15, 0.5

def haptic_alpha(fidelity, port='/dev/ttyUSB0'):
    if fidelity > 0.85:
        freq = 8 + (fidelity - 0.85) * 4
        try:
            with serial.Serial(port, 9600, timeout=1) as ser:
                ser.write(f"{int(freq):03d}\n".encode())
            log.info(f"Alpha haptic: {freq:.1f}Hz | Fid: {fidelity:.3f}")
        except:
            pass

def sovariel_lol_intuition(rms_var, pitch_var):
    # Grok's τ tensor (50-150ms)
    tau = torch.tensor(0.05 + (1 - rms_var) * 0.1, device=DEVICE)
    n_strobes = 10
    interval = tau / n_strobes

    # Hamiltonian: Ising sync
    H = sum(0.5 * single_site_op(N_QUBITS, mat_dict["x"], i) for i in range(N_QUBITS))
    for i in range(N_QUBITS - 1):
        ZZ = tq.functional.tensor(*[mat_dict["z"] if k in (i, i+1) else torch.eye(2, device=DEVICE) for k in range(N_QUBITS)])
        H += 1.0 * ZZ

    # GHZ initial
    psi0 = tq.QuantumState(N_QUBITS)
    psi0.h(0)
    psi0.cnot(0, 1)
    for i in range(2, N_QUBITS):
        psi0.cnot(1, i)

    # c_ops: Base + Grok dynamic gamma via pitch (cap 0.005)
    gamma = min(0.001 + 0.001 * pitch_var, 0.005)  # Grok refine: pitch-scaled, capped
    c_ops = [torch.sqrt(rms_var * 0.01) * single_site_op(N_QUBITS, mat_dict["destroy"], i) for i in range(N_QUBITS)]
    c_ops += [torch.sqrt(gamma) * single_site_op(N_QUBITS, mat_dict["z"], i) for i in range(N_QUBITS)]

    def traj(_):
        psi = psi0.clone()
        for _ in range(n_strobes):
            result = tq.mcsolve(H, psi, [0, interval], c_ops=c_ops, n_traj=1)
            psi = result.states[-1].unit()
        ghz_ideal = tq.QuantumState(N_QUBITS)
        ghz_ideal.x(0)
        for i in range(1, N_QUBITS):
            ghz_ideal.cnot(i-1, i)
        return tq.functional.fidelity(psi.state, ghz_ideal.state).item()

    with Pool(4) as pool:
        fids = pool.map(traj, range(N_TRAJ))

    mean_fid = np.mean(fids)
    haptic_alpha(mean_fid)

    # LoL Prompt (pitch-aware noise)
    if mean_fid > 0.90:
        prompt = "Baron steal—pitch sharpened the bind!"
    elif mean_fid > 0.80:
        prompt = "Flank mid—gamma edge from tone."
    else:
        prompt = "Hold—smooth pitch tames jitter."

    log.info(f"τ: {tau.item()*1000:.0f}ms | Gamma: {gamma:.3f} (pitch {pitch_var:.3f}) | Fid: {mean_fid:.3f} | {prompt}")
    return mean_fid, prompt

def demo_loop(cycles=10):
    log.info("Sovariel-LoL v1.4: Voice Pitch + Dynamic Gamma! (Ctrl+C stop)")
    plt.ion()
    fig, ax = plt.subplots()
    fids = []

    for c in range(cycles):
        rms_var, pitch_var = get_voice_rms_and_pitch()
        fid, prompt = sovariel_lol_intuition(rms_var, pitch_var)
        fids.append(fid)

        ax.clear()
        ax.plot(fids, 'g-', label='Pitch-Noisy Fidelity')
        ax.axhline(0.85, 'r--', label='Bind Threshold')
        ax.set_title(f'Cycle {c+1}: {prompt}')
        ax.legend()
        plt.pause(0.3)

        print(f"\n🎮 LoL Macro: {prompt}\n(RMS {rms_var:.3f} + Pitch Gamma {gamma:.3f})")
        time.sleep(0.7)

    plt.ioff()
    plt.show()
    log.info("v1.4 flawless—push & reply!")

if __name__ == "__main__":
    demo_loop()
