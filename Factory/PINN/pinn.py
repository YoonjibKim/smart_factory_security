import numpy as np
import pandas as pd
import os
import torch
from torch import nn
from torch import optim


class PINN:
    def __init__(self):
        self.device = None
        self.model = None
        self.optimizer = None

    def _load_dataset(self, train_data_path, test_data_path):
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

        # 🚨 [수정됨] 입력이 3개(t, x, OP)로 늘어남!
        self.model = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _filter_features(self, train_df, test_df):
        raise NotImplementedError("이 함수는 상속받은 장비 클래스에서 개별적으로 구현해야 합니다.")

    def _compute_physics_loss(self, t, x, op):  # OP 파라미터 추가
        raise NotImplementedError("이 함수는 상속받은 장비 클래스에서 개별적으로 구현해야 합니다.")

    def _train_pinn(self, train_df, target_cols, op_cols, epochs=10000, sample_ratio=0.1, patience=100, min_delta=1e-6):
        print(f"[*] PINN 학습 시작 (총 {epochs} Epochs)")

        mse_loss_fn = nn.MSELoss()

        # 1. 하이브리드 샘플링 (Max 변화량 기반)
        df_sample = self.__hybrid_sampling(train_df, target_cols, sample_ratio)
        df_sample = df_sample.ffill().fillna(0)

        # 2. 시간(t) 텐서 생성
        t_values_raw = df_sample.index.values.astype(float)
        t_values_norm = t_values_raw / (t_values_raw.max() if t_values_raw.max() > 0 else 1.0)
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        # 3. 공간 확장 (Spatial Stacking): 1존과 2존 데이터를 분리하여 세로로 병합
        x_zone1 = torch.full((len(df_sample), 1), 1.0, dtype=torch.float32).to(self.device)
        op_zone1 = torch.tensor(df_sample[op_cols[0]].values, dtype=torch.float32).view(-1, 1).to(self.device)
        u_zone1 = torch.tensor(df_sample[target_cols[0]].values, dtype=torch.float32).view(-1, 1).to(self.device)

        x_zone2 = torch.full((len(df_sample), 1), 2.0, dtype=torch.float32).to(self.device)
        op_zone2 = torch.tensor(df_sample[op_cols[1]].values, dtype=torch.float32).view(-1, 1).to(self.device)
        u_zone2 = torch.tensor(df_sample[target_cols[1]].values, dtype=torch.float32).view(-1, 1).to(self.device)

        # [Concat] 위아래로 이어 붙이기
        t_data = torch.cat([t_tensor, t_tensor], dim=0)
        x_data = torch.cat([x_zone1, x_zone2], dim=0)
        op_data = torch.cat([op_zone1, op_zone2], dim=0)
        u_true = torch.cat([u_zone1, u_zone2], dim=0)

        # 정규화
        u_true = u_true / 100.0
        op_max = op_data.max() if op_data.max() > 0 else 1.0
        op_data = op_data / op_max

        # Physics Loss용 텐서
        t_phys = t_data.clone().detach().requires_grad_(True)
        x_phys = x_data.clone().detach().requires_grad_(True)
        op_phys = op_data.clone().detach()

        # 🚨 [조기 종료를 위한 변수 세팅]
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # [A] Data Loss
            inputs_data = torch.cat([t_data, x_data, op_data], dim=1)
            u_pred = self.model(inputs_data)
            loss_data = mse_loss_fn(u_pred, u_true)

            # [B] Physics Loss
            loss_physics = self._compute_physics_loss(t_phys, x_phys, op_phys)

            # [C] Total Loss 및 역전파
            total_loss = loss_data + loss_physics
            total_loss.backward()
            self.optimizer.step()

            # 🚨 [조기 종료 확인 로직]
            current_loss = total_loss.item()

            # Loss가 우리가 정한 기준(min_delta) 이상으로 좋아졌다면?
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0  # 카운터 초기화
                best_model_state = self.model.state_dict()  # 💡 최고 성능일 때의 뇌(가중치) 상태 백업!
            else:
                patience_counter += 1  # 안 좋아졌으면 인내심 카운터 1 증가

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | "
                      f"Data Loss: {loss_data.item():.4e} | "
                      f"Phys Loss: {loss_physics.item():.4e} | "
                      f"Total: {total_loss.item():.4e}")

            # 🚨 [조기 종료 발동 조건]
            if patience_counter >= patience:
                print(f"\n[*] 🛑 조기 종료(Early Stopping) 발동! (Epoch {epoch + 1})")
                print(f"    - 이유: {patience} Epoch 동안 Total Loss가 유의미하게 개선되지 않음.")
                print(f"    - 최적 Total Loss: {best_loss:.4e}")

                # 백업해둔 최고 상태로 롤백 (이게 진짜 중요해!)
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                    print("    - 💡 모델 가중치를 최적의 상태로 복원했습니다.")
                break

        print("\n[*] 모델 학습이 완료되었습니다.")

    def _test_pinn(self, test_df, target_cols, op_cols):
        print(f"\n[*] 테스트 데이터 전체({len(test_df)}개)로 평가 시작...")

        self.model.eval()

        # 🚨 [수정됨] 10,000개 샘플링 제거하고 test_df 전체를 그대로 사용!
        df_test = test_df.ffill().fillna(0).copy()

        t_values_raw = df_test.index.values.astype(float)
        t_values_norm = t_values_raw / (t_values_raw.max() if t_values_raw.max() > 0 else 1.0)
        t_tensor = torch.tensor(t_values_norm, dtype=torch.float32).view(-1, 1).to(self.device)

        x_zone1 = torch.full((len(df_test), 1), 1.0, dtype=torch.float32).to(self.device)
        op_zone1 = torch.tensor(df_test[op_cols[0]].values, dtype=torch.float32).view(-1, 1).to(self.device)
        u_zone1 = torch.tensor(df_test[target_cols[0]].values, dtype=torch.float32).view(-1, 1).to(self.device)

        x_zone2 = torch.full((len(df_test), 1), 2.0, dtype=torch.float32).to(self.device)
        op_zone2 = torch.tensor(df_test[op_cols[1]].values, dtype=torch.float32).view(-1, 1).to(self.device)
        u_zone2 = torch.tensor(df_test[target_cols[1]].values, dtype=torch.float32).view(-1, 1).to(self.device)

        t_test = torch.cat([t_tensor, t_tensor], dim=0)
        x_test = torch.cat([x_zone1, x_zone2], dim=0)
        op_test = torch.cat([op_zone1, op_zone2], dim=0)
        u_true = torch.cat([u_zone1, u_zone2], dim=0)

        u_true = u_true / 100.0
        op_max = op_test.max() if op_test.max() > 0 else 1.0
        op_test = op_test / op_max

        # 기울기(Gradient) 연산을 꺼서 평가 속도와 VRAM 사용량을 대폭 최적화
        with torch.no_grad():
            inputs_test = torch.cat([t_test, x_test, op_test], dim=1)
            u_pred = self.model(inputs_test)

            mse_loss_fn = nn.MSELoss()
            test_loss = mse_loss_fn(u_pred, u_true)

        real_temperature_error = torch.sqrt(test_loss) * 100.0

        print(f"[*] Test MSE Loss (스케일링 됨): {test_loss.item():.4e}")
        print(f"[*] 실제 예측 오차: 평균 약 {real_temperature_error.item():.2f}°C 차이 발생\n")

    def __hybrid_sampling(self, df, target_cols, sample_ratio=0.1, uniform_ratio=0.2):
        """
        자식 클래스에서 지정한 다중 타겟(target_cols)의 변화량을 기준으로 하이브리드 샘플링을 수행합니다.
        - sample_ratio: 전체 데이터 중 몇 %를 추출할 것인지 (1.0 이상이면 전체 데이터 반환)
        - uniform_ratio: 추출할 데이터 중 몇 %를 등간격으로 뽑을 것인지 (기본 20%)
        """
        total_rows = len(df)

        # 🚨 [추가된 프리패스 로직] sample_ratio가 1.0(100%) 이상이면 샘플링 없이 전체 데이터를 그대로 반환!
        if sample_ratio >= 1.0:
            print(f"[*] 샘플링 비율 100% 설정됨. 하이브리드 샘플링을 생략하고 {total_rows}개 전체를 사용합니다.")
            return df.copy()

        # 1. 최종 추출할 목표 개수 계산
        target_samples = max(1, int(total_rows * sample_ratio))

        # 2. 등간격과 변화량 추출 개수 분할
        num_uniform = int(target_samples * uniform_ratio)
        num_gradient = target_samples - num_uniform

        # 3. 등간격 샘플링 (전체 흐름 파악용)
        uniform_indices = np.linspace(0, total_rows - 1, num_uniform, dtype=int)

        # 4. 변화량 기반 샘플링 (물리적 변곡점 파악용)
        combined_diff = np.zeros(total_rows)

        # 다중 변수의 정규화된 "최대값(Max)"을 추출
        for col in target_cols:
            numeric_series = pd.to_numeric(df[col], errors='coerce').fillna(0)
            diff = numeric_series.diff().abs().fillna(0)

            max_diff = diff.max()
            if max_diff > 0:
                diff_norm = diff / max_diff
            else:
                diff_norm = diff

            # 합산이 아닌 np.maximum을 사용하여 가장 가파른 변화량을 우선시
            combined_diff = np.maximum(combined_diff, diff_norm.values)

        # 0으로 나누어지는 것을 방지
        combined_diff += 1e-6

        # 확률 분포로 변환
        prob_dist = combined_diff / combined_diff.sum()

        # 확률 분포에 따라 랜덤 추출 (변화가 큰 곳이 뽑힐 확률이 높음)
        gradient_indices = np.random.choice(
            df.index,
            size=num_gradient,
            replace=False,
            p=prob_dist
        )

        # 5. 인덱스 통합 및 정렬 (시간 순서대로 학습해야 하므로 오름차순 정렬 필수)
        combined_indices = np.unique(np.concatenate([uniform_indices, gradient_indices]))
        combined_indices.sort()

        df_sampled = df.loc[combined_indices].copy()

        # np.unique 때문에 중복 제거되어 최종 개수가 num_uniform + num_gradient 보다 조금 작을 수 있음
        print(f"[*] 다중 하이브리드 샘플링 완료 (Max 기준)")
        print(f"    - 총 {total_rows}개 중 {len(df_sampled)}개 추출 완료")

        return df_sampled