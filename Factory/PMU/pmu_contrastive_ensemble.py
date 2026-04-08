import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. 딥러닝 모델 아키텍처 정의 (PyTorch)
# ==========================================
class ContrastiveEnsembleNet(nn.Module):
    """
    PMU 데이터와 PINN(물리 기반) 특징을 결합하여 오동작을 탐지하는 대조 앙상블 모델
    """

    def __init__(self, input_dim):
        super(ContrastiveEnsembleNet, self).__init__()
        # PMU 데이터를 처리하는 경로
        self.pmu_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # PINN(물리 법칙) 특징을 추출하는 경로
        self.pinn_branch = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        # 두 경로를 결합하여 오동작 여부 판별 (1: 이상, 0: 정상)
        self.classifier = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        pmu_feat = self.pmu_branch(x)
        pinn_feat = self.pinn_branch(x)
        # 대조 학습을 위해 추출된 특징들을 이어 붙임 (Concatenate)
        combined = torch.cat((pmu_feat, pinn_feat), dim=1)
        return self.classifier(combined), pmu_feat, pinn_feat


# ==========================================
# 2. 메인 앙상블 클래스 구현
# ==========================================
class PmuContrastiveEnsemble:
    def __init__(self, heat_treatment_train_dataset_path, heat_treatment_test_dataset_path, pmu_data_dir_path):
        # 원본 데이터 로드
        self.__pinn_train_df = self.__load_csv_to_df(heat_treatment_train_dataset_path)
        self.__pinn_test_df = self.__load_csv_to_df(heat_treatment_test_dataset_path)
        self.__pmu_data_dict = self.__load_all_csv_to_dict(pmu_data_dir_path)

        # 내부 상태 저장용 딕셔너리
        self.__prepared_train_datasets = {}
        self.__prepared_test_datasets = {}
        self.models = {}  # 학습된 모델 인스턴스들을 저장

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
            raise NotADirectoryError(f"지정된 경로가 디렉토리가 아닙니다: {dir_path}")

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
                    df = pd.read_csv(file_path, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='utf-8')

                csv_dict[dict_key] = df
                print(f"Successfully loaded into dict: {file_name} (Rows: {len(df)})")

            except Exception as e:
                print(f"[!] '{file_name}' 읽기 중 오류 발생: {e}")

        return csv_dict

    def prepare_datasets(self, machine_configs: dict):
        """
        공정별 타겟 컬럼 딕셔너리를 받아 데이터셋을 분리하고 지정된 경로에 CSV로 저장합니다.

        :param machine_configs: {"DRYING_1": [컬럼리스트], "HARDENING": [컬럼리스트], ...}
        :return: (train_dict, test_dict)
        """
        save_dir = os.path.join("Factory", "PMU", "Dataset")
        os.makedirs(save_dir, exist_ok=True)

        for machine_name, target_cols in machine_configs.items():
            if not target_cols:
                print(f"[*] {machine_name}: 타겟 컬럼이 없어 건너뜁니다.")
                continue

            # 존재하는 컬럼만 안전하게 필터링
            valid_train_cols = [c for c in target_cols if c in self.__pinn_train_df.columns]
            valid_test_cols = [c for c in target_cols if c in self.__pinn_test_df.columns]

            # 데이터 추출 (깊은 복사)
            train_subset = self.__pinn_train_df[valid_train_cols].copy()
            test_subset = self.__pinn_test_df[valid_test_cols].copy()

            # CSV 파일 경로 설정
            train_csv_path = os.path.join(save_dir, f"{machine_name}_train.csv")
            test_csv_path = os.path.join(save_dir, f"{machine_name}_test.csv")

            # CSV 저장 (한글 깨짐 방지)
            train_subset.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
            test_subset.to_csv(test_csv_path, index=False, encoding='utf-8-sig')

            print(f"[V] Saved: {train_csv_path} / {test_csv_path}")

            # 딕셔너리 내부 저장
            self.__prepared_train_datasets[machine_name] = train_subset
            self.__prepared_test_datasets[machine_name] = test_subset

        return self.__prepared_train_datasets, self.__prepared_test_datasets

    def generate_model(self, target_columns_dict: dict):
        """
        데이터를 불러와 GPU/CPU 환경에 맞게 학습시키고,
        테스트 데이터로 성능을 평가한 뒤 모델을 저장합니다.
        """
        # 시스템 GPU 가용성 체크
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[System] 현재 설정된 연산 디바이스: {device.type.upper()}")

        model_save_dir = os.path.join("Factory", "PMU", "Model")
        dataset_dir = os.path.join("Factory", "PMU", "Dataset")
        os.makedirs(model_save_dir, exist_ok=True)

        for machine_name in target_columns_dict.keys():
            print(f"\n{'=' * 50}")
            print(f"[*] '{machine_name}' 모델 학습 및 평가 시작...")
            print(f"{'=' * 50}")

            train_path = os.path.join(dataset_dir, f"{machine_name}_train.csv")
            test_path = os.path.join(dataset_dir, f"{machine_name}_test.csv")

            try:
                # 결측치(NaN)를 0으로 채움
                train_df = self.__load_csv_to_df(train_path).fillna(0)
                test_df = self.__load_csv_to_df(test_path).fillna(0)
            except FileNotFoundError:
                print(f"[!] '{machine_name}'의 데이터를 찾을 수 없어 건너뜁니다.")
                continue

            # -----------------------------------------------------
            # [수정 필요] 데이터셋 내 정답(Label) 컬럼명 지정
            # (실제 데이터셋의 라벨 컬럼 이름으로 'label'을 변경하세요)
            # -----------------------------------------------------
            target_col = 'label'

            if target_col in train_df.columns:
                # [수정된 부분] 숫자형 데이터만 선택 (.select_dtypes)
                X_train_df = train_df.drop(columns=[target_col]).select_dtypes(include=['number'])
                y_train = train_df[target_col].values
                X_test_df = test_df.drop(columns=[target_col]).select_dtypes(include=['number'])
                y_test = test_df[target_col].values
            else:
                print(f"[!] 경고: '{target_col}' 컬럼이 없습니다. 테스트용 더미 라벨(전부 정상)로 임시 진행합니다.")
                # [수정된 부분] 숫자형 데이터만 선택 (.select_dtypes)
                X_train_df = train_df.select_dtypes(include=['number'])
                y_train = torch.zeros(len(train_df), 1).numpy()
                X_test_df = test_df.select_dtypes(include=['number'])
                y_test = torch.zeros(len(test_df), 1).numpy()

            # values 추출
            X_train = X_train_df.values
            X_test = X_test_df.values

            # 텐서 변환 및 디바이스(GPU/CPU) 할당
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            input_dim = X_train_tensor.shape[1]
            if input_dim == 0:
                print(f"[!] 입력 차원이 0입니다 (숫자형 데이터가 없음). 건너뜁니다.")
                continue

            # 모델 생성 및 디바이스 할당
            model = ContrastiveEnsembleNet(input_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()  # 이진 분류 손실 함수

            # -------------------------
            # 1. 학습 (Training Phase)
            # -------------------------
            model.train()
            epochs = 20  # 반복 횟수 (필요에 따라 조절)

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
            # 2. 성능 평가 (Test Phase)
            # -------------------------
            model.eval()
            print("  [Test Phase]")
            with torch.no_grad():
                test_output, _, _ = model(X_test_tensor)
                test_loss = criterion(test_output, y_test_tensor)

                # 예측값 산출 (0.5 이상이면 이상(1), 미만이면 정상(0))
                predictions = (test_output >= 0.5).float()

                # 정확도(Accuracy) 계산
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