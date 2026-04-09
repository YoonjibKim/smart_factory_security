import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import onnxruntime as ort
from Factory.PINN.pinn import PINN

# 불필요한 PyTorch/ONNX 워닝(경고) 메시지 숨김 처리
warnings.filterwarnings("ignore")


# ==========================================
# 1. 병렬 학습 앙상블 모델 (PyTorch)
# ==========================================
class ContrastiveEnsembleNet(nn.Module):
    """
    PMU 데이터와 PINN(물리 법칙) 특징을 결합하여 예측을 수행하는 모델 앙상블 클래스
    """

    def __init__(self, input_dim):
        super(ContrastiveEnsembleNet, self).__init__()
        # PMU 데이터 특징 추출기
        self.pmu_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # PINN(물리 법칙) 특징 추출기
        self.pinn_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 두 특징을 결합하여 최종 분류 (1: 정상, 0: 불량)
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pmu_feat = self.pmu_branch(x)
        pinn_feat = self.pinn_branch(x)
        # 두 특징을 채널 차원 기준으로 병합 (Concatenate)
        combined = torch.cat((pmu_feat, pinn_feat), dim=1)
        return self.classifier(combined), pmu_feat, pinn_feat


# ==========================================
# 2. 모델 학습용 앙상블 클래스
# ==========================================
class PmuContrastiveEnsemble(PINN):
    def __init__(self, heat_treatment_train_dataset_path, heat_treatment_test_dataset_path, pmu_data_dir_path):
        PINN.__init__(self)
        # 기본 데이터셋 로드
        self.__pinn_train_df = self.__load_csv_to_df(heat_treatment_train_dataset_path)
        self.__pinn_test_df = self.__load_csv_to_df(heat_treatment_test_dataset_path)
        self.__pmu_data_dict = self.__load_all_csv_to_dict(pmu_data_dir_path)

        # 모델 학습 데이터셋 딕셔너리
        self.__prepared_train_datasets = {}
        self.__prepared_test_datasets = {}
        self.models = {}  # 설비별 학습 파라미터 저장

    def __load_csv_to_df(self, file_path: str):  # noqa
        """
        주어진 경로의 CSV 파일을 읽어 DataFrame으로 반환합니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding='cp949')  # noqa
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8')  # noqa

        print(f"Successfully loaded: {os.path.basename(file_path)} (Rows: {len(df)})")  # noqa
        return df

    def __load_all_csv_to_dict(self, dir_path: str):  # noqa
        """
        주어진 디렉토리 내의 모든 CSV 파일을 읽어 딕셔너리로 반환합니다.
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {dir_path}")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"주어진 경로가 디렉토리가 아닙니다: {dir_path}")

        csv_dict = {}
        file_list = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        if not file_list:
            print(f"[*] 주의: '{dir_path}' 내에 CSV 파일이 없습니다.")
            return csv_dict

        for file_name in file_list:
            file_path = os.path.join(dir_path, file_name)
            dict_key = os.path.splitext(file_name)[0]

            try:
                try:
                    df = pd.read_csv(file_path, encoding='cp949') # noqa
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='utf-8') # noqa

                csv_dict[dict_key] = df
                print(f"Successfully loaded into dict: {file_name} (Rows: {len(df)})") # noqa

            except Exception as e:
                print(f"[!] '{file_name}' 읽기 중 오류 발생: {e}")

        return csv_dict

    def prepare_datasets(self, machine_configs: dict):
        """
        설비별 대상 칼럼 정보(machine_configs)를 바탕으로 데이터셋을 구성하여 별도의 CSV로 저장합니다.

        :param machine_configs: {"DRYING_1": [칼럼리스트], "HARDENING": [칼럼리스트], ...}
        :return: (train_dict, test_dict)
        """
        save_dir = os.path.join("Factory", "PMU", "Dataset")
        os.makedirs(save_dir, exist_ok=True)

        for machine_name, target_cols in machine_configs.items():
            if not target_cols:
                print(f"[*] {machine_name}: 대상 칼럼이 비어 있습니다.")
                continue

            # 데이터프레임 내에 존재하는 칼럼만 필터링
            valid_train_cols = [c for c in target_cols if c in self.__pinn_train_df.columns]
            valid_test_cols = [c for c in target_cols if c in self.__pinn_test_df.columns]

            # 데이터 추출 (깊은 복사)
            train_subset = self.__pinn_train_df[valid_train_cols].copy()
            test_subset = self.__pinn_test_df[valid_test_cols].copy()

            # CSV 저장 경로 구성
            train_csv_path = os.path.join(save_dir, f"{machine_name}_train.csv")
            test_csv_path = os.path.join(save_dir, f"{machine_name}_test.csv")

            # CSV 저장 (한글 깨짐 방지)
            train_subset.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
            test_subset.to_csv(test_csv_path, index=False, encoding='utf-8-sig')

            print(f"[V] Saved: {train_csv_path} / {test_csv_path}")

            # 딕셔너리에 객체 할당
            self.__prepared_train_datasets[machine_name] = train_subset
            self.__prepared_test_datasets[machine_name] = test_subset

        return self.__prepared_train_datasets, self.__prepared_test_datasets

    def generate_model(self, target_columns_dict: dict):
        """
        추출된 데이터셋을 바탕으로 GPU/CPU 디바이스 할당 후 앙상블 모델을 학습시키고,
        학습이 완료되면 가중치 정보를 지정된 경로에 저장합니다.
        """
        # 사용 가능한 GPU 디바이스 확인
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[System] 현재 학습에 할당된 디바이스: {device.type.upper()}")

        model_save_dir = os.path.join("Factory", "PMU", "Model")
        dataset_dir = os.path.join("Factory", "PMU", "Dataset")
        os.makedirs(model_save_dir, exist_ok=True)

        for machine_name in target_columns_dict.keys():
            print(f"\n{'=' * 50}")
            print(f"[*] '{machine_name}' 모델 학습 및 평가 진행 중...")
            print(f"{'=' * 50}")

            train_path = os.path.join(dataset_dir, f"{machine_name}_train.csv")
            test_path = os.path.join(dataset_dir, f"{machine_name}_test.csv")

            try:
                # 결측치(NaN)는 0으로 치환
                train_df = self.__load_csv_to_df(train_path).fillna(0)
                test_df = self.__load_csv_to_df(test_path).fillna(0)
            except FileNotFoundError:
                print(f"[!] '{machine_name}'의 데이터셋을 찾을 수 없어 건너뜁니다.")
                continue

            # -----------------------------------------------------
            # [변수 분리] 입력특성(X)과 타겟(Label) 데이터 분리
            # (현재 샘플에서는 임시로 타겟 칼럼명을 'label'로 가정합니다)
            # -----------------------------------------------------
            target_col = 'label'

            if target_col in train_df.columns:
                # [문자열 제외] 수치형 데이터만 추출 (.select_dtypes)
                X_train_df = train_df.drop(columns=[target_col]).select_dtypes(include=['number'])
                y_train = train_df[target_col].values
                X_test_df = test_df.drop(columns=[target_col]).select_dtypes(include=['number'])
                y_test = test_df[target_col].values
            else:
                print(f"[!] 주의: '{target_col}' 칼럼이 없습니다. 테스트를 위해 임의의 타겟(전부 0)을 임시 생성합니다.")
                # [문자열 제외] 수치형 데이터만 추출 (.select_dtypes)
                X_train_df = train_df.select_dtypes(include=['number'])
                y_train = torch.zeros(len(train_df), 1).numpy()
                X_test_df = test_df.select_dtypes(include=['number'])
                y_test = torch.zeros(len(test_df), 1).numpy()

            # values 추출
            X_train = X_train_df.values
            X_test = X_test_df.values

            # 모델 입력용 텐서 변환 및 디바이스(GPU/CPU) 할당
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            input_dim = X_train_tensor.shape[1]
            if input_dim == 0:
                print(f"[!] 입력 차원이 0입니다 (수치형 데이터가 없음). 건너뜁니다.")
                continue

            # 모델 초기화 및 최적화기 설정
            model = ContrastiveEnsembleNet(input_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()  # 이진 분류 손실 함수

            # -------------------------
            # 1. 학습 진행 (Training Phase)
            # -------------------------
            model.train()
            epochs = 20  # 학습 반복 횟수 (테스트용 임시 설정)

            print("  [Train Phase]")
            for epoch in range(epochs):
                optimizer.zero_grad()
                output, _, _ = model(X_train_tensor)

                loss = criterion(output, y_train_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0:
                    print(f"   - Epoch {epoch + 1:02d}/{epochs} | Loss: {loss.item():.4f}")

            # -------------------------
            # 2. 모델 평가 (Test Phase)
            # -------------------------
            model.eval()
            print("  [Test Phase]")
            with torch.no_grad():
                test_output, _, _ = model(X_test_tensor)
                test_loss = criterion(test_output, y_test_tensor)

                # 정확도 계산 (0.5 이상이면 정상(1), 미만이면 불량(0))
                predictions = (test_output >= 0.5).float()

                # 정답률(Accuracy) 산출
                correct = (predictions == y_test_tensor).sum().item()
                accuracy = correct / len(y_test_tensor) * 100

                print(f"   - Test Loss: {test_loss.item():.4f}")
                print(f"   - Accuracy : {accuracy:.2f}%")

            # -------------------------
            # 3. 모델 저장 (.pth)
            # -------------------------
            model_path = os.path.join(model_save_dir, f"{machine_name}_model.pth")
            torch.save(model.state_dict(), model_path)

            self.models[machine_name] = model
            print(f"\n[V] '{machine_name}' 모델 저장 완료: {model_path}")

        return self.models

    def generate_edge_model(self, target_columns_dict: dict):
        """
        저장된 PyTorch 모델(.pth)을 로드하여 Edge 디바이스용 ONNX 모델로 변환 및 저장합니다.
        평가 시 가변적인 배치 크기(Batch Size)를 받을 수 있도록 dynamic_axes를 설정합니다.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[System] Edge 모델 변환을 위한 디바이스: {device.type.upper()}")

        dataset_dir = os.path.join("Factory", "PMU", "Dataset")
        base_model_dir = os.path.join("Factory", "PMU", "Model")
        edge_model_dir = os.path.join(base_model_dir, "Edge_Model")
        os.makedirs(edge_model_dir, exist_ok=True)

        for machine_name in target_columns_dict.keys():
            print(f"\n{'=' * 50}")
            print(f"[*] '{machine_name}' Edge 모델(ONNX) 생성 준비 중...")
            print(f"{'=' * 50}")

            pth_model_path = os.path.join(base_model_dir, f"{machine_name}_model.pth")
            if not os.path.exists(pth_model_path):
                print(f"[!] '{machine_name}'의 저장된 PyTorch 모델(.pth)이 없습니다: {pth_model_path}")
                continue

            train_path = os.path.join(dataset_dir, f"{machine_name}_train.csv")
            try:
                train_df = self.__load_csv_to_df(train_path).fillna(0)
                target_col = 'label'

                if target_col in train_df.columns:
                    X_train_df = train_df.drop(columns=[target_col]).select_dtypes(include=['number'])
                else:
                    X_train_df = train_df.select_dtypes(include=['number'])

                input_dim = X_train_df.shape[1]
                if input_dim == 0:
                    print(f"[!] '{machine_name}'의 입력 차원(input_dim)이 0입니다. 건너뜁니다.")
                    continue
            except Exception as e:
                print(f"[!] '{machine_name}' 데이터셋 로드 및 input_dim 확인 실패: {e}")
                continue

            try:
                model = ContrastiveEnsembleNet(input_dim).to(device)
                model.load_state_dict(torch.load(pth_model_path, map_location=device))
                model.eval()
                print(f"[*] '{machine_name}' PyTorch 모델 로드 성공 (input_dim: {input_dim})")
            except Exception as e:
                print(f"[!] '{machine_name}' 모델 로드 실패: {e}")
                continue

            edge_model_path = os.path.join(edge_model_dir, f"{machine_name}_edge_model.onnx")

            # 더미 입력은 여전히 배치 사이즈 1로 생성하지만, 변환 시 dynamic_axes로 가변 설정함
            dummy_input = torch.randn(1, input_dim).to(device)

            try:
                torch.onnx.export(
                    model,
                    dummy_input, # noqa
                    edge_model_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input_features'],
                    output_names=['output_class', 'pmu_feat', 'pinn_feat'],
                    # 입력과 출력의 0번째 차원(Batch Size)을 가변적으로 설정
                    dynamic_axes={
                        'input_features': {0: 'batch_size'},
                        'output_class': {0: 'batch_size'},
                        'pmu_feat': {0: 'batch_size'},
                        'pinn_feat': {0: 'batch_size'}
                    }
                )
                print(f"[V] '{machine_name}' Edge 모델 변환 및 저장 완료: {edge_model_path}")
            except Exception as e:
                print(f"[!] '{machine_name}' Edge 모델 변환 실패: {e}")

        # 모든 머신의 ONNX 변환이 끝난 후 자동으로 평가 호출
        print("\n[System] 모든 Edge 모델(ONNX) 생성이 완료되었습니다. 자동 평가를 시작합니다.")
        self.__evaluate_edge_model(target_columns_dict)

    def __evaluate_edge_model(self, target_columns_dict: dict):
        """
        저장된 Edge 모델(ONNX)을 onnxruntime으로 로드하여
        테스트 데이터셋에 대한 정확도(Accuracy)와 추론 속도(Latency)를 평가합니다.
        """
        dataset_dir = os.path.join("Factory", "PMU", "Dataset")
        edge_model_dir = os.path.join("Factory", "PMU", "Model", "Edge_Model")

        for machine_name in target_columns_dict.keys():
            print(f"\n{'=' * 50}")
            print(f"[*] '{machine_name}' Edge 모델(ONNX) 성능 평가 중...")
            print(f"{'=' * 50}")

            edge_model_path = os.path.join(edge_model_dir, f"{machine_name}_edge_model.onnx")
            test_path = os.path.join(dataset_dir, f"{machine_name}_test.csv")

            if not os.path.exists(edge_model_path):
                print(f"[!] ONNX 모델이 존재하지 않습니다: {edge_model_path}")
                continue

            try:
                test_df = self.__load_csv_to_df(test_path).fillna(0)
                # 평가에서도 임의로 부여된 label과 동일한 기준으로 채점
                target_col = 'label'

                if target_col in test_df.columns:
                    X_test = test_df.drop(columns=[target_col]).select_dtypes(include=['number']).values
                    y_test = test_df[target_col].values
                else:
                    X_test = test_df.select_dtypes(include=['number']).values
                    y_test = np.zeros(len(test_df))

                X_test = X_test.astype(np.float32)
                y_test = y_test.astype(np.float32)

                # ONNX 런타임 세션 로드
                session = ort.InferenceSession(edge_model_path)
                input_name = session.get_inputs()[0].name

                # Inference 시간 측정 시작
                start_time = time.time()
                ort_outs = session.run(None, {input_name: X_test})
                inference_time = time.time() - start_time

                # 결과 산출 (Sigmoid 값 기준 0.5 이상이면 1, 아니면 0)
                predictions = (ort_outs[0] >= 0.5).astype(np.float32).flatten()
                correct = (predictions == y_test).sum()
                accuracy = (correct / len(y_test)) * 100

                # ---------------- [분포 확인 코드 추가됨] ----------------
                unique_y, counts_y = np.unique(y_test, return_counts=True)
                unique_p, counts_p = np.unique(predictions, return_counts=True)
                # ---------------------------------------------------------

                print(f"  [Evaluate Result]")
                print(f"   - ONNX 모델 로드 : {os.path.basename(edge_model_path)}")
                print(f"   - 데이터 샘플 수 : {len(X_test)} 개")
                print(f"   - 실제 정답 분포 : {dict(zip(unique_y, counts_y))}")  # 정답의 정상/불량 개수
                print(f"   - 모델 예측 분포 : {dict(zip(unique_p, counts_p))}")  # 모델이 찍은 정상/불량 개수
                print(f"   - Inference Time : {inference_time:.4f} 초")
                print(f"   - Accuracy       : {accuracy:.2f}%\n")

            except Exception as e:
                print(f"[!] '{machine_name}' Edge 모델 평가 중 오류 발생: {e}")