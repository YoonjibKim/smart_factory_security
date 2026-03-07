import os
import torch
from Factory.PINN.pinn import PINN


class Washing(PINN):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)

        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path

        self.washing_alpha = 0.01

        self._washing_target_columns = [
            'TAG_MIN',
            '세정기'
        ]

        # 🚨 PINN.py가 두 번째 타겟(Zone2)을 찾는 하드코딩을 우회하기 위해
        # 가짜 타겟(Dummy Zone)을 하나 추가합니다.
        self._washing_temp_cols = [
            '세정기',
            '세정_Dummy_Zone_2'
        ]

        # OP도 동일하게 2개를 유지합니다.
        self._washing_op_cols = ['세정_Dummy_OP_1', '세정_Dummy_OP_2']

    def _operate_washing(self):  # noqa
        print("세정기 동작")

    def _filter_washing_features(self, train_df, test_df):
        try:
            train_filtered = train_df[self._washing_target_columns].copy()
            test_filtered = test_df[self._washing_target_columns].copy()

            # Dummy OP 추가
            train_filtered['세정_Dummy_OP_1'] = 0.0
            train_filtered['세정_Dummy_OP_2'] = 0.0

            test_filtered['세정_Dummy_OP_1'] = 0.0
            test_filtered['세정_Dummy_OP_2'] = 0.0

            # 🚨 Dummy 타겟(Zone) 추가
            train_filtered['세정_Dummy_Zone_2'] = 0.0
            test_filtered['세정_Dummy_Zone_2'] = 0.0

            print(f"[*] 세정기 데이터 필터링 완료: {train_filtered.columns.tolist()}")
            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 오타 확인 필요: {e}")
            return train_df, test_df

    def _build_washing(self, train_dataset_path, test_dataset_path):
        print("세정기 설치 및 PINN 초기화")

        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, filtered_test_df = self._filter_washing_features(train_df, test_df)

        # 모델 구조 초기화
        self._make_model()

        # 🚨 경로 자동 정리: 넘어온 경로가 폴더일 경우 파일명 강제 지정
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(
                self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "washing_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        # 🚨 train_pinn_flag에 따른 조건부 학습 및 저장/로드 로직
        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 모델 학습을 시작합니다.")
            self._train_pinn(
                train_df=filtered_train_df,
                target_cols=self._washing_temp_cols,
                op_cols=self._washing_op_cols,
                epochs=10000,
                sample_ratio=0.5
            )

            self._test_pinn(filtered_test_df, self._washing_temp_cols, self._washing_op_cols)

            # 모델 저장 로직
            save_dir = os.path.dirname(actual_model_path)
            if save_dir:  # 경로에 폴더가 포함되어 있다면 폴더 생성
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), actual_model_path)
            print(f"[*] 학습 완료: 모델이 성공적으로 저장되었습니다 -> {actual_model_path}")

        else:
            # 학습을 하지 않을 경우 저장된 모델 불러오기
            print(f"[*] 추론 모드 작동: 저장된 모델을 불러옵니다.")
            if os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path))
                self.model.eval()  # 평가 모드로 전환
                print(f"[*] 모델 로드 성공 -> {actual_model_path}")
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다 -> {actual_model_path}")

    def _compute_physics_loss(self, t, x, op):
        inputs = torch.cat([t, x, op], dim=1)

        # u의 shape: (batch_size, 2) -> 진짜 세정기 타겟 1개 + 가짜 Dummy 타겟 1개
        u = self.model(inputs)

        total_physics_loss = 0.0

        # for문이 2번 돌면서 진짜 타겟과 가짜 타겟 모두에 대해 물리 법칙을 계산합니다.
        for i in range(u.shape[1]):
            u_zone = u[:, i:i + 1]

            u_t = torch.autograd.grad(u_zone, t, grad_outputs=torch.ones_like(u_zone), create_graph=True)[0]
            u_x = torch.autograd.grad(u_zone, x, grad_outputs=torch.ones_like(u_zone), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

            f = u_t - self.washing_alpha * u_xx
            total_physics_loss += torch.mean(f ** 2)

        return total_physics_loss / u.shape[1]