import os
import numpy as np
import onnxruntime as ort


class AttackDetection:
    def __init__(self, model_dir="Factory/PMU/Model/Edge_Model"):
        self.model_dir = model_dir
        self.sessions = {}
        self.model_mapping = {
            "건조로(Drying)": "MachineType.DRYING_edge_model.onnx",
            "가열로(Hardening)": "MachineType.HARDENING_edge_model.onnx",
            "솔트배스(Salt Bath)": "MachineType.SALT_BATH_edge_model.onnx",
            "세척기(Washing)": "MachineType.WASHING_edge_model.onnx"
        }
        self._load_models()

    def _load_models(self):
        """프로세스 시작 시 모델을 메모리에 로드합니다."""
        print("\n[백그라운드] 에지 PINN 방어 엔진 가동 중...")
        for machine_name, file_name in self.model_mapping.items():
            model_path = os.path.join(self.model_dir, file_name)
            if os.path.exists(model_path):
                try:
                    # CPU 가속을 사용하여 독립 프로세스에서 실행
                    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    self.sessions[machine_name] = session
                except Exception:
                    pass

    def detect(self, machine_name, sensor_data):
        if machine_name not in self.sessions or not sensor_data:
            return None

        try:
            input_data = np.array(sensor_data, dtype=np.float32)
            if input_data.ndim == 1:
                input_data = np.expand_dims(input_data, axis=0)

            session = self.sessions[machine_name]
            input_name = session.get_inputs()[0].name
            expected_dim = session.get_inputs()[0].shape[-1]

            # 자동 차원 치유 (지난 답변 적용)
            if input_data.shape[-1] != expected_dim:
                if input_data.shape[-1] > expected_dim:
                    input_data = input_data[:, -expected_dim:]
                else:
                    padding = np.zeros((input_data.shape[0], expected_dim - input_data.shape[-1]), dtype=np.float32)
                    input_data = np.hstack((padding, input_data))

            outputs = session.run(None, {input_name: input_data})
            result_val = np.array(outputs[0]).flatten()[0]

            # 임계값 기반 이진 탐지
            threshold = 5.0
            if result_val > threshold:
                print(f"\n🚨 [실시간 탐지] {machine_name} 공격 감지! 오차: {result_val:.4f}")
                return True
            return False
        except Exception:
            return None