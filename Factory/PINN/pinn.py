import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch import optim
import copy


class PINN:
    def __init__(self):
        self.device = None
        self.model = None
        self.optimizer = None

    def _load_dataset(self, train_data_path, test_data_path):  # noqa
        if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
            raise FileNotFoundError("데이터 경로를 확인해주세요.")

        train_df = pd.read_csv(train_data_path, encoding='cp949')
        test_df = pd.read_csv(test_data_path, encoding='cp949')

        print(f"Dataset Loaded: Train({len(train_df)}), Test({len(test_df)})")
        return train_df, test_df

    def _make_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 모델 학습 Device: {self.device}")

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

        u_true = u_true / 1000.0
        TRAIN_OP_MAX = 100.0
        op_data = op_data / TRAIN_OP_MAX

        t_phys = t_data.clone().detach().requires_grad_(True)
        x_phys = x_data.clone().detach().requires_grad_(True)
        op_phys = op_data.clone().detach()

        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            inputs_data = torch.cat([t_data, x_data, op_data], dim=1)
            u_pred = self.model(inputs_data)
            loss_data = mse_loss_fn(u_pred, u_true)

            loss_physics = self._compute_physics_loss(t_phys, x_phys, op_phys)

            total_loss = loss_data + loss_physics
            total_loss.backward()
            self.optimizer.step()

            current_loss = total_loss.item()

            if best_loss == float('inf') or current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | Data Loss: {loss_data.item():.4e} | Phys Loss: {loss_physics.item():.4e} | Total: {total_loss.item():.4e}")

            if patience_counter >= patience:
                print(f"\n[*] 🛑 조기 종료(Early Stopping) 발동! (Epoch {epoch + 1})")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                break

        if patience_counter < patience and best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print("\n[*] 모델 학습이 완료되었습니다.")

    def _test_pinn(self, test_df, target_cols, op_cols):
        print(f"\n[*] 테스트 데이터 전체({len(test_df)}개)로 평가 시작...")

        self.model.eval()
        df_test = test_df.ffill().fillna(0).copy()

        TRAIN_T_MAX = 2353798.0
        t_values_raw = df_test.index.values.astype(float)
        t_values_norm = t_values_raw / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        t_data_list = []
        x_data_list = []
        op_data_list = []
        u_true_list = []

        for i in range(len(target_cols)):
            zone_idx = float(i + 1)
            x_zone = torch.full((len(df_test), 1), zone_idx, dtype=torch.float32).to(self.device)
            op_zone = torch.tensor(df_test[op_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)
            u_zone = torch.tensor(df_test[target_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)

            t_data_list.append(t_tensor)
            x_data_list.append(x_zone)
            op_data_list.append(op_zone)
            u_true_list.append(u_zone)

        t_test = torch.cat(t_data_list, dim=0)
        x_test = torch.cat(x_data_list, dim=0)
        op_test = torch.cat(op_data_list, dim=0)
        u_true = torch.cat(u_true_list, dim=0)

        u_true = u_true / 1000.0
        TRAIN_OP_MAX = 100.0
        op_test = op_test / TRAIN_OP_MAX

        with torch.no_grad():
            inputs_test = torch.cat([t_test, x_test, op_test], dim=1)
            u_pred = self.model(inputs_test)

            mse_loss_fn = nn.MSELoss()
            test_loss = mse_loss_fn(u_pred, u_true)

        real_temperature_error = torch.sqrt(test_loss) * 1000.0

        print(f"[*] Test MSE Loss (스케일링 됨): {test_loss.item():.4e}")
        print(f"[*] 실제 예측 오차: 평균 약 {real_temperature_error.item():.2f}°C 차이 발생\n")

    # =========================================================================
    # 🚀 신규 기능 1: 공정 최적화 시뮬레이션 (What-If 테스트)
    # =========================================================================
    def _simulate_what_if_op(self, test_df, target_cols, op_cols, target_op_col, adjustment_value=0):
        """
        특정 OP(가스 밸브 등) 값을 인위적으로 조작하여 온도가 어떻게 변하는지 가상 테스트합니다. (- ~ +)
        """
        if test_df is None:
            return

        print(f"\n[🧪 What-If 시뮬레이션] '{target_op_col}' 값을 {adjustment_value} 만큼 조작하여 가상 테스트를 진행합니다.")

        # 원본 데이터를 망가뜨리지 않기 위해 깊은 복사 수행
        simulated_df = test_df.copy()

        if target_op_col in simulated_df.columns:
            # 설정한 값만큼 해당 OP 컬럼의 모든 센서 데이터를 조작
            simulated_df[target_op_col] = simulated_df[target_op_col] + adjustment_value

            # 기존 _test_pinn 채점관에게 조작된 시험지를 제출하여 결과를 확인!
            self._test_pinn(simulated_df, target_cols, op_cols)
        else:
            print(f"[!] 에러: 데이터에 '{target_op_col}' 컬럼이 없어 시뮬레이션을 중단합니다.")

    # =========================================================================
    # 🚨 신규 기능 2: 이상 탐지 시스템 (Anomaly Detection)
    # =========================================================================
    def _detect_anomalies(self, test_df, target_cols, op_cols, threshold_celsius=15.0):
        """
        AI가 예측한 '정상 온도'와 '실제 센서 온도'의 차이가 설정된 임계치(예: 15도) 이상
        벌어지면 센서 고장이나 가스 누출 등 물리적 이상을 감지하여 알람을 발생시킵니다.
        """
        if test_df is None:
            return

        print(f"\n[🚨 이상 탐지 시스템 가동] 정상 모델 온도 대비 {threshold_celsius}°C 이상 차이나는 위험 구간 탐색 중...")

        self.model.eval()
        df_test = test_df.ffill().fillna(0).copy()

        TRAIN_T_MAX = 2353798.0
        t_values_raw = df_test.index.values.astype(float)
        t_values_norm = t_values_raw / TRAIN_T_MAX
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        t_data_list, x_data_list, op_data_list, u_true_list = [], [], [], []

        for i in range(len(target_cols)):
            zone_idx = float(i + 1)
            x_zone = torch.full((len(df_test), 1), zone_idx, dtype=torch.float32).to(self.device)
            op_zone = torch.tensor(df_test[op_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)
            u_zone = torch.tensor(df_test[target_cols[i]].values, dtype=torch.float32).view(-1, 1).to(self.device)

            t_data_list.append(t_tensor)
            x_data_list.append(x_zone)
            op_data_list.append(op_zone)
            u_true_list.append(u_zone)

        t_test = torch.cat(t_data_list, dim=0)
        x_test = torch.cat(x_data_list, dim=0)
        op_test = torch.cat(op_data_list, dim=0)
        u_true = torch.cat(u_true_list, dim=0)

        # 모델 스케일링 복원용 세팅
        u_true_scaled = u_true / 1000.0
        op_test_scaled = op_test / 100.0

        with torch.no_grad():
            inputs_test = torch.cat([t_test, x_test, op_test_scaled], dim=1)
            u_pred_scaled = self.model(inputs_test)

        # 실제 온도(°C)로 복원하여 절대 오차 계산
        u_pred_real = u_pred_scaled * 1000.0
        u_true_real = u_true_scaled * 1000.0

        errors = torch.abs(u_pred_real - u_true_real)

        # 임계치를 넘는 이상치 탐색
        anomaly_mask = errors > threshold_celsius
        anomaly_count = anomaly_mask.sum().item()
        total_count = errors.numel()

        print(f"[*] 스캔 완료: 총 {total_count}개의 데이터 지점 중 {anomaly_count}개의 이상(Anomaly) 신호 발견.")

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

        combined_diff += 1e-6
        prob_dist = combined_diff / combined_diff.sum()
        gradient_indices = np.random.choice(df.index, size=num_gradient, replace=False, p=prob_dist)

        combined_indices = np.unique(np.concatenate([uniform_indices, gradient_indices]))
        combined_indices.sort()
        return df.loc[combined_indices].copy()

    def _get_pinn_model(self):
        return self.model