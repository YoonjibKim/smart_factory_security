import torch
import io
import onnxruntime as ort # ONNX 모델 로드를 위해 필요

class EdgeModel:
    def __init__(self):
        pass

    def _convert_onnx_model(self, pytorch_model):
        # 1. 파일 경로 대신 메모리 버퍼를 생성합니다.
        onnx_buffer = io.BytesIO()

        # 2. 더미 입력 생성 (건조로 모델은 t, x, op 3개의 입력이 concat 되어 들어감)
        # 차원: (Batch Size=1, Features=3)
        dummy_input = torch.randn(1, 3).to(next(pytorch_model.parameters()).device)

        pytorch_model.eval()

        # 3. PyTorch -> ONNX 변환 (파일 경로 대신 버퍼 객체 전달)
        try:
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_buffer,
                export_params=True,
                opset_version=14,  # 🚨 11 -> 14로 상향 조정 (버전 충돌/Gemm 에러 해결)
                do_constant_folding=True,
                input_names=['input_t_x_op'],
                output_names=['output_temp']
            )
            print("[*] Edge용 ONNX 모델 메모리 변환 성공")

            # 버퍼에 담긴 ONNX 모델의 바이너리(bytes) 데이터를 리턴합니다.
            return onnx_buffer.getvalue()

        except Exception as e:
            print(f"[!] Edge 모델 변환 실패: {e}")
            return None

    # 🚨 누락되었던 에지 모델 로드 전용 함수 추가
    def _load_model(self, edge_model_path):
        """저장된 에지 모델(ONNX)을 onnxruntime 세션으로 로드합니다."""
        try:
            # ONNX 런타임을 사용해 에지 모델 세션 생성
            edge_session = ort.InferenceSession(edge_model_path)
            print(f"[*] Edge 모델(ONNX) 전용 런타임 메모리 적재 성공")
            return edge_session
        except Exception as e:
            print(f"[!] Edge 모델 로드 실패: {e}")
            return None