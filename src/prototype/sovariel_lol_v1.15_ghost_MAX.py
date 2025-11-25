# sovariel_lol_v1.15_ghost_MAX_adaptive_pro.py
# Final demo version – 12 PM CST 26 Nov 2025
# © 2025 AgapeIntelligence — MIT License

import mss, numpy as np, torch, torchquantum as tq
from torchquantum.functional import h, cz, rz
from torchquantum.mitigation import ZNEMitigator
import cv2, time, json, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SovarielPro")

# ------------------- Quantum Agent (unchanged logic, minor speedups) -------------------
class AdaptiveFiAgent(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 5
        self.q_device = tq.QuantumDevice(n_wires=5, bsz=1)
        self.mitigator = ZNEMitigator(scale_factors=[1.0, 1.5, 3.0], degree=2)
        self.q_device.noise_model = {
            "name": "amplitude_damping",
            "gate_time": 0.00004,
            "t1": 2.4,
            "t2": 1.6
        }

    def set_shots(self, beta: float):
        shots = 4096 if beta >= 0.85 else \
                8192 if beta >= 0.70 else \
                16384 if beta >= 0.55 else 32768
        self.measure = tq.MeasureAll(tq.PauliZ, shots=shots)

    @tq.static_support
    def forward(self, beta_proxy: float):
        self.set_shots(beta_proxy)
        self.q_device.reset_states(1)

        h(self.q_device, wires=[0])
        for i in range(4):
            cz(self.q_device, wires=[i, i+1])
        h(self.q_device, wires=[1, 2, 3, 4])

        noise = max(0.004, 0.07 * (1 - beta_proxy))
        for w in range(5):
            rz(self.q_device, wires=w, params=torch.tensor([noise * np.pi]))

        raw = self.measure(self.q_device)
        return self.mitigator.mitigate(raw).mean()


# ------------------- Fast, reliable dynamic ROI -------------------
sct = mss.mss()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Fallback ROIs (centered on 1920×1080 primary monitor)
FALLBACK_EYE   = {"top": 180, "left": 560, "width": 800, "height": 240}
FALLBACK_MOUTH = {"top": 540, "left": 640, "width": 640, "height": 200}

agent = AdaptiveFiAgent()
full_monitor = sct.monitors[1]  # primary gaming monitor

while True:
    t0 = time.time()
    try:
        # 1. Grab full screen once (cheap) for face detection
        screen = np.array(sct.grab(full_monitor))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120))

        if len(faces) > 0:
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]  # biggest face
            eye_roi   = {"top": y + 20,           "left": x + w//6, "width": w*5//6, "height": int(h*0.35)}
            mouth_roi = {"top": y + int(h*0.62),  "left": x + w//8, "width": w*3//4, "height": int(h*0.35)}
        else:
            eye_roi, mouth_roi = FALLBACK_EYE, FALLBACK_MOUTH

        # 2. Grab only the two small ROIs
        eye_img   = np.array(sct.grab(eye_roi))
        mouth_img = np.array(sct.grab(mouth_roi))

        # 3. RMS motion proxy (frame-to-frame diff)
        eye_rms   = np.sqrt(np.mean(np.diff(eye_img.astype(np.int16), axis=0)**2))
        mouth_rms = np.sqrt(np.mean(np.diff(mouth_img.astype(np.int16), axis=0)**2))
        beta = np.clip((eye_rms + mouth_rms) / 42.0, 0.0, 1.0)

        # 4. Quantum decision
        fi = agent(beta).item()
        decision = "AGGRESSIVE" if fi > 0.965 else "HOLD"

        payload = {
            "ts": int(time.time()*1000),
            "beta": round(beta, 4),
            "Fi": round(fi, 6),
            "shots": agent.measure.shots,
            "decision": decision,
            "lat_ms": int((time.time() - t0)*1000)
        }
        print(json.dumps(payload))

    except Exception as e:
        log.warning(f"Frame dropped: {e}")
        print(json.dumps({"ts": int(time.time()*1000), "decision": "HOLD", "lat_ms": int((time.time()-t0)*1000)}))

    # Target ~60 Hz max (never exceed 250 ms rule)
    elapsed = time.time() - t0
    time.sleep(max(0.005, 0.058 - elapsed))
