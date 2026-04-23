import os
import time
import csv
import numpy as np
import onnxruntime as ort
from pyModbusTCP.client import ModbusClient


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

        # Modbus HMI 레지스터 감시용
        self.modbus_client = ModbusClient(host='127.0.0.1', port=5020, auto_open=True)
        self.addr_map = {
            "건조로(Drying)": 120, "가열로(Hardening)": 130,
            "솔트배스(Salt Bath)": 150, "세척기(Washing)": 110
        }

        # HMI와 공유하기 위해 공용 경로(/tmp/) 사용
        self.log_path = "/tmp/prediction_log.csv"
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'is_anomaly'])

        self._load_models()

    def _load_models(self):
        print("\n[백그라운드] 에지 PMU 상태 감시 엔진 가동 중...")
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

        is_cyber_attack = False
        is_physical_anomaly = False

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

                # 공격 시그니처 확인
                if any(atk in decoded for atk in ["ERR", "DOS", "MODIFIED", "OLD"]):
                    is_cyber_attack = True

        # 2. Physical 관점: PINN 추론
        if machine_name in self.sessions:
            try:
                input_data = np.array(sensor_data, dtype=np.float32)
                if input_data.ndim == 1: input_data = np.expand_dims(input_data, axis=0)
                session = self.sessions[machine_name]
                outputs = session.run(None, {session.get_inputs()[0].name: input_data})
                if outputs[0].flatten()[0] > 5.0: is_physical_anomaly = True
            except Exception:
                pass

        final_anomaly = 1 if (is_cyber_attack or is_physical_anomaly) else 0

        # 로그 기록 (공용 경로)
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), final_anomaly])

        if final_anomaly:
            print(f"🚨 [감시] {machine_name} 이상 탐지!")
        return bool(final_anomaly)