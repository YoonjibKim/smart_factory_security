import os
import time
import csv
import numpy as np
import onnxruntime as ort
from pyModbusTCP.client import ModbusClient
from Honey_Pot.honey_pot import HoneyPot


class PMUStateMonitor:
    def __init__(self, model_dir="Factory/PMU/Model/Edge_Model"):
        self.model_dir = model_dir
        self.sessions = {}
        self.model_mapping = {
            "건조로(Drying)": "MachineType.DRYING_edge_model.onnx",
            "가열로(Hardening)": "MachineType.HARDENING_edge_model.onnx",
            "솔트배스(Salt Bath)": "MachineType.SALT_BATH_edge_model.onnx",
            "세척기(Washing)": "MachineType.WASHING_edge_model.onnx"
        }

        self.modbus_client = ModbusClient(host='127.0.0.1', port=5020, auto_open=True)
        self.addr_map = {
            "건조로(Drying)": 120, "가열로(Hardening)": 130,
            "솔트배스(Salt Bath)": 150, "세척기(Washing)": 110
        }

        self.log_path = "/tmp/prediction_log.csv"
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'predicted_type'])

        self.honeypot = HoneyPot()
        self.previous_state = {machine: "Normal" for machine in self.model_mapping.keys()}

        self._load_models()

    def _load_models(self):
        print("\n[백그라운드] 에지 PMU 상태 감시 엔진 가동 중 (다중 분류 적용)...")
        for machine_name, file_name in self.model_mapping.items():
            model_path = os.path.join(self.model_dir, file_name)
            if os.path.exists(model_path):
                try:
                    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    self.sessions[machine_name] = session
                except Exception:
                    pass

    def detect(self, machine_name, sensor_data):
        if not sensor_data: return None

        predicted_type = "Normal"

        # 1. Cyber 관점: Modbus 레지스터 감시
        modbus_addr = self.addr_map.get(machine_name)
        if modbus_addr:
            regs = self.modbus_client.read_holding_registers(modbus_addr, 10)
            if regs:
                byte_array = bytearray()
                for val in regs:
                    byte_array.append((val >> 8) & 0xFF)
                    byte_array.append(val & 0xFF)
                decoded = byte_array.decode('utf-8', errors='ignore').rstrip('\x00')

                # 구체적인 공격 이름 분류
                if "DOS" in decoded:
                    predicted_type = "DoS"
                elif "MODIFIED" in decoded:
                    predicted_type = "MITM"
                elif "ERR" in decoded:
                    predicted_type = "FDIA"
                elif "OLD" in decoded:
                    predicted_type = "Replay"

        # 2. Physical 관점: PINN 추론
        if predicted_type == "Normal" and machine_name in self.sessions:
            try:
                input_data = np.array(sensor_data, dtype=np.float32)
                if input_data.ndim == 1: input_data = np.expand_dims(input_data, axis=0)
                session = self.sessions[machine_name]
                outputs = session.run(None, {session.get_inputs()[0].name: input_data})

                if outputs[0].flatten()[0] > 100000.0:
                    predicted_type = "Replay"  # 물리적 이상은 Replay성 위협으로 매핑
            except Exception:
                pass

        current_time = time.time()
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([current_time, predicted_type])

        # 상태가 변했을 때만 로그 출력 및 허니팟 호출
        last_state = self.previous_state.get(machine_name, "Normal")
        if predicted_type != "Normal" and predicted_type != last_state:
            print(f"\n🚨 [감시] {machine_name} 정상 흐름 중 위협({predicted_type}) 포착! (허니팟 라우팅)")
            self.honeypot.trap_threat(
                machine_name=machine_name,
                attack_type=predicted_type,
                timestamp=current_time
            )

        self.previous_state[machine_name] = predicted_type
        return predicted_type != "Normal"