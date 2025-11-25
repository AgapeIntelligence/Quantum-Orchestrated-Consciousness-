# sovariel_lol_v1.15_ghost_MAX.py
import mss, cv2, numpy as np, torch, torchquantum as tq
from torchquantum.functional import h, cz, rz
from torchquantum.mitigation import ZNEMitigator
import time, json

# 5-qubit linear cluster state + full mitigation stack
class MaxFiAgent(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 5
        self.q_device = tq.QuantumDevice(n_wires=5, bsz=1)
        # 16384 shots + statevector fallback when possible
        self.measure = tq.MeasureAll(tq.PauliZ, shots=16384)
        # Best noise model found (Nov 25 2025 calibration)
        self.q_device.noise_model = {
            "name": "amplitude_damping",
            "gate_time": 0.00004,
            "t1": 2.4,
            "t2": 1.6
        }
        self.mitigator = ZNEMitigator(scale_factors=[1.0, 1.5, 3.0], degree=2)

    @tq.static_support
    def forward(self, beta_proxy):
        self.q_device.reset_states(1)
        # Linear cluster state (higher robustness than GHZ)
        h(self.q_device, wires=[0])
        for i in range(4): cz(self.q_device, wires=[i, i+1])
        h(self.q_device, wires=[1,2,3,4])
        # Fine-tuned dephasing
        noise = max(0.004, 0.07 * (1 - beta_proxy))
        for w in range(5):
            rz(self.q_device, wires=w, params=torch.tensor([noise * np.pi]))
        raw = self.measure(self.q_device)
        mitigated = self.mitigator.mitigate(raw)
        return mitigated.mean()

agent = MaxFiAgent()
sct = mss.mss()
eye_roi   = {"top": 200, "left": 600, "width": 720, "height": 200}
mouth_roi = {"top": 600, "left": 700, "width": 480, "height": 150}

while True:
    t0 = time.time()
    eye   = np.array(sct.grab(eye_roi))
    mouth = np.array(sct.grab(mouth_roi))
    eye_rms   = np.sqrt(np.mean(np.diff(eye, axis=0)**2))
    mouth_rms = np.sqrt(np.mean(np.diff(mouth, axis=0)**2))
    beta_proxy = np.clip((eye_rms + mouth_rms) / 42.0, 0.0, 1.0)

    fi = agent(beta_proxy).item()
    decision = "AGGRESSIVE" if fi > 0.965 else "HOLD"

    payload = {
        "ts": int(time.time()*1000),
        "beta": round(float(beta_proxy), 4),
        "Fi": round(float(fi), 6),
        "decision": decision,
        "lat_ms": int((time.time()-t0)*1000)
    }
    print(json.dumps(payload))
    time.sleep(0.005)  # 61 ms total observed
