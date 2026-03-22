import numpy as np
import pandas as pd
import os
import torch
import copy
from torch import nn
from torch import optim
from Factory.Edge_Model.edge_model import EdgeModel


class PINN(EdgeModel):
    def __init__(self):
        self.device = None
        self.model = None
        self.optimizer = None
        EdgeModel.__init__(self)

    def _load_dataset(self, train_data_path, test_data_path):  # noqa
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            raise FileNotFoundError("데이터 경로를 확인해주세요.")

        train_df = pd.read_csv(train_data_path, encoding='cp949')
        test_df = pd.read_csv(test_data_path, encoding='cp949')

        print(f"Dataset Loaded: Train({len(train_df)}), Test({len(test_df)})")
        return train_df, test_df

    def _convert_to_edge_model(self, model):
        return self._convert_onnx_model(model)

    def _make_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 모델 학습 Device: {self.device}")

        # Dummy tensor for device initialization
        torch.zeros(1).to(self.device)

        self.model = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _filter_features(self, train_df, test_df):
        raise NotImplementedError("이 함수는 상속받은 장비 클래스에서 개별적으로 구현해야 합니다.")

    def _compute_physics_loss(self, t, x, op):
        raise NotImplementedError("이 함수는 상속받은 장비 클래스에서 개별적으로 구현해야 합니다.")

    def _predict_with_model(self, inputs):
        """
        PyTorch와 Edge 모델(ONNX)을 모두 지원하는 통합 추론 래퍼
        """
        # 1. Edge 모델(ONNX)이 로드되어 있는 경우 (운영/테스트 단계)
        if getattr(self, 'edge_session', None) is not None:
            input_name = self.edge_session.get_inputs()[0].name
            input_shape = self.edge_session.get_inputs()[0].shape

            # ONNX 모델이 기대하는 배치 사이즈 확인 (예: 1)
            # shape이 [1, 3] 등일 때 첫 번째 값이 배치 사이즈. 'batch_size' 같은 동적 문자열이면 None 처리
            expected_batch = input_shape[0] if len(input_shape) > 0 and isinstance(input_shape[0], int) else None

            # 텐서를 numpy로 변환
            if isinstance(inputs, torch.Tensor):
                inputs_np = inputs.detach().cpu().numpy().astype('float32')
            else:
                inputs_np = inputs.astype('float32')

            actual_batch = inputs_np.shape[0]

            # 배치 사이즈가 같거나, 모델이 동적 배치(Dynamic Shape)를 지원하면 한 번에 추론
            if expected_batch is None or expected_batch == actual_batch:
                pred_np = self.edge_session.run(None, {input_name: inputs_np})[0]
            else:
                # 🌟 [수정된 부분] 모델이 고정 배치(예: 1)를 요구하는데 입력 데이터가 더 큰 경우 쪼개서 추론
                preds = []
                for i in range(0, actual_batch, expected_batch):
                    batch_data = inputs_np[i: i + expected_batch]

                    # 마지막 배치가 expected_batch보다 작을 경우 패딩(Padding) 처리
                    pad_len = expected_batch - batch_data.shape[0]
                    if pad_len > 0:
                        batch_data = np.pad(batch_data, ((0, pad_len), (0, 0)), mode='edge')

                    batch_pred = self.edge_session.run(None, {input_name: batch_data})[0]

                    # 패딩된 부분은 제외하고 유효한 결과만 저장
                    if pad_len > 0:
                        batch_pred = batch_pred[:-pad_len]

                    preds.append(batch_pred)

                # 쪼개서 추론한 결과를 다시 하나로 합침
                pred_np = np.concatenate(preds, axis=0)

            # 결과값을 다시 PyTorch Tensor로 변환하여 후처리 코드와 호환 유지
            target_device = self.device if self.device is not None else torch.device('cpu')
            return torch.tensor(pred_np).to(target_device)

        # 2. PyTorch 모델이 로드되어 있는 경우 (학습/빌드 단계)
        elif getattr(self, 'model', None) is not None:
            self.model.eval()
            with torch.no_grad():
                return self.model(inputs)

        # 3. 모델이 로드되지 않은 경우 예외 처리
        else:
            raise RuntimeError("[!] 에러: 추론을 수행할 모델(PyTorch 또는 ONNX)이 로드되지 않았습니다.")

    def _train_pinn(self, train_df, target_cols, op_cols, epochs=10000, sample_ratio=0.1, patience=100, min_delta=1e-5):
        print(f"[*] PINN 학습 시작 (총 {epochs} Epochs)")

        mse_loss_fn = nn.MSELoss()

        df_sample = self.__hybrid_sampling(train_df, target_cols, sample_ratio)
        df_sample = df_sample.ffill().fillna(0)

        TRAIN_T_MAX = 2353798.0
        t_values_raw = df_sample.index.values.astype(float)
        t_values_norm = t_values_raw / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        t_data_list = []
        x_data_list = []
        op_data_list = []
        u_true_list = []

        for i in range(len(target_cols)):
            zone_idx = float(i + 1)
            x_zone = torch.full((len(df_sample), 1), zone_idx, dtype=torch.float32).to(self.device)
            op_zone = torch.tensor(df_sample[op_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)
            u_zone = torch.tensor(df_sample[target_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)

            t_data_list.append(t_tensor)
            x_data_list.append(x_zone)
            op_data_list.append(op_zone)
            u_true_list.append(u_zone)

        t_data = torch.cat(t_data_list, dim=0)
        x_data = torch.cat(x_data_list, dim=0)
        op_data = torch.cat(op_data_list, dim=0)
        u_true = torch.cat(u_true_list, dim=0)

        t_data.requires_grad_(True)
        x_data.requires_grad_(True)
        op_data.requires_grad_(True)

        best_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            inputs = torch.cat([t_data, x_data, op_data], dim=1)
            u_pred = self.model(inputs)

            data_loss = mse_loss_fn(u_pred, u_true)
            physics_loss = self._compute_physics_loss(t_data, x_data, op_data)

            total_loss = data_loss + physics_loss
            total_loss.backward()
            self.optimizer.step()

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Loss: {total_loss.item():.6f} (Data: {data_loss.item():.6f}, Physics: {physics_loss.item():.6f})")

            # Early Stopping Check
            if best_loss - total_loss.item() > min_delta:
                best_loss = total_loss.item()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"[*] Early stopping triggered at epoch {epoch}")
                break

        print("[*] PINN 학습 완료.")

    def _test_pinn(self, test_df, target_cols, op_cols):
        print(f"[*] 테스트 데이터 전체({len(test_df)}개)로 평가 시작...")

        if self.device is None:
            self.device = torch.device("cpu")

        test_df = test_df.ffill().fillna(0)

        TRAIN_T_MAX = 2353798.0
        t_values_raw = test_df.index.values.astype(float)
        t_values_norm = t_values_raw / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        total_mse = 0.0

        for i in range(len(target_cols)):
            zone_idx = float(i + 1)
            x_zone = torch.full((len(test_df), 1), zone_idx, dtype=torch.float32).to(self.device)
            op_zone = torch.tensor(test_df[op_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)
            u_true = torch.tensor(test_df[target_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)

            inputs = torch.cat([t_tensor, x_zone, op_zone], dim=1)

            # ONNX 래퍼가 자동으로 데이터 쪼개서 추론 처리함
            u_pred = self._predict_with_model(inputs)

            mse = torch.mean((u_pred - u_true) ** 2).item()
            total_mse += mse
            print(f"  - {target_cols[i]} Test MSE: {mse:.4f}")

        print(f"[*] 전체 평균 Test MSE: {total_mse / len(target_cols):.4f}\n")

    def _simulate_what_if_op(self, test_df, target_cols, op_cols, test_op_col):
        print(f"[*] What-If 시뮬레이션: {test_op_col} 조작에 따른 온도 변화 예측")

        if self.device is None:
            self.device = torch.device("cpu")

        sample_df = test_df.iloc[:100].copy()

        TRAIN_T_MAX = 2353798.0
        t_values_norm = sample_df.index.values.astype(float) / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        zone_idx = float(op_cols.index(test_op_col) + 1)
        x_zone = torch.full((len(sample_df), 1), zone_idx, dtype=torch.float32).to(self.device)

        original_op = torch.tensor(sample_df[test_op_col].values, dtype=torch.float32).view(-1, 1).to(self.device)
        inputs_orig = torch.cat([t_tensor, x_zone, original_op], dim=1)

        pred_orig = self._predict_with_model(inputs_orig)

        modified_op = original_op + 10.0
        inputs_mod = torch.cat([t_tensor, x_zone, modified_op], dim=1)

        pred_mod = self._predict_with_model(inputs_mod)

        diff = (pred_mod - pred_orig).mean().item()
        print(f"  -> {test_op_col} 값을 +10 증가시켰을 때, 예상 온도 변화량 (평균): {diff:.4f}\n")

    def _detect_anomalies(self, test_df, target_cols, op_cols, threshold=5.0):
        print(f"[*] 물리 법칙 기반 이상 탐지 (Anomaly Detection) 시작 (Threshold: {threshold})")

        if self.device is None:
            self.device = torch.device("cpu")

        test_df = test_df.ffill().fillna(0)

        TRAIN_T_MAX = 2353798.0
        t_values_norm = test_df.index.values.astype(float) / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        anomaly_count = 0

        for i in range(len(target_cols)):
            zone_idx = float(i + 1)
            x_zone = torch.full((len(test_df), 1), zone_idx, dtype=torch.float32).to(self.device)
            op_zone = torch.tensor(test_df[op_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)
            u_true = torch.tensor(test_df[target_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)

            inputs = torch.cat([t_tensor, x_zone, op_zone], dim=1)

            u_pred = self._predict_with_model(inputs)

            error = torch.abs(u_pred - u_true)
            anomalies = (error > threshold).sum().item()
            anomaly_count += anomalies

        print(f"  -> 총 {len(test_df) * len(target_cols)}개의 데이터 포인트 중 {anomaly_count}개의 이상(Anomaly) 신호 발견.")

        if anomaly_count > 0:
            print(f"[!] ⚠️ 경고: 즉각적인 설비 점검 필요 (센서 고장 또는 밸브/열원 누출 의심)")
        else:
            print(f"[*] ✅ 현재 공정의 센서와 물리 환경은 완벽하게 일치하며 정상 작동 중입니다.\n")

    def __hybrid_sampling(self, df, target_cols, sample_ratio=0.1, uniform_ratio=0.2):  # noqa
        total_rows = len(df)
        if sample_ratio >= 1.0:
            return df.copy()

        target_samples = max(1, int(total_rows * sample_ratio))
        num_uniform = int(target_samples * uniform_ratio)
        num_gradient = target_samples - num_uniform

        uniform_indices = np.linspace(0, total_rows - 1, num_uniform, dtype=int)
        combined_diff = np.zeros(total_rows)

        for col in target_cols:
            numeric_series = pd.to_numeric(df[col], errors='coerce').fillna(0)
            diff = numeric_series.diff().abs().fillna(0)
            max_diff = diff.max()
            diff_norm = diff / max_diff if max_diff > 0 else diff
            combined_diff = np.maximum(combined_diff, diff_norm.values)

        gradient_probs = combined_diff / combined_diff.sum() if combined_diff.sum() > 0 else np.ones(
            total_rows) / total_rows
        gradient_indices = np.random.choice(total_rows, size=num_gradient, replace=False, p=gradient_probs)

        final_indices = np.unique(np.concatenate([uniform_indices, gradient_indices]))
        return df.iloc[final_indices].copy()