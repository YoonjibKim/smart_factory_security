import torch
from Factory.PINN.pinn import PINN


class SaltBath(PINN):
    def __init__(self):
        PINN.__init__(self)
        self.salt_bath_alpha = 0.01

        self._salt_bath_target_columns = [
            'TAG_MIN',
            '솔트조 온도 1 Zone',
            '솔트조 온도 2 Zone'
        ]

        self._salt_bath_temp_cols = [
            '솔트조 온도 1 Zone',
            '솔트조 온도 2 Zone'
        ]

        # 🚨 [수정됨] PINN.py 에러 방지를 위해 가짜(Dummy) OP 컬럼 이름 지정
        # (온도 Zone 개수에 맞춰 2개를 만들어 줍니다)
        self._salt_bath_op_cols = ['솔트_Dummy_OP_1', '솔트_Dummy_OP_2']

    def _operate_salt_bath(self):  # noqa
        print("솔트조 동작")

    def _filter_salt_bath_features(self, train_df, test_df):
        try:
            train_filtered = train_df[self._salt_bath_target_columns].copy()
            test_filtered = test_df[self._salt_bath_target_columns].copy()

            # 🚨 [추가됨] PINN.py를 속이기 위해 값이 0인 가짜 OP 컬럼을 추가
            train_filtered['솔트_Dummy_OP_1'] = 0.0
            train_filtered['솔트_Dummy_OP_2'] = 0.0

            test_filtered['솔트_Dummy_OP_1'] = 0.0
            test_filtered['솔트_Dummy_OP_2'] = 0.0

            print(f"[*] 솔트조 데이터 필터링 완료: {train_filtered.columns.tolist()}")
            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 오타 확인 필요: {e}")
            return train_df, test_df

    def _build_salt_bath(self, train_dataset_path, test_dataset_path):
        print("솔트조 설치 및 PINN 초기화")

        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, filtered_test_df = self._filter_salt_bath_features(train_df, test_df)

        self._make_model()

        self._train_pinn(
            train_df=filtered_train_df,
            target_cols=self._salt_bath_temp_cols,
            op_cols=self._salt_bath_op_cols,  # 이제 빈 리스트가 아니라 가짜 OP 2개가 들어갑니다.
            epochs=10000,
            sample_ratio=0.5
        )

        self._test_pinn(filtered_test_df, self._salt_bath_temp_cols, self._salt_bath_op_cols)

    def _compute_physics_loss(self, t, x, op):
        # 🚨 [수정됨] 차원 불일치 에러를 막기 위해 op를 반드시 포함시킵니다!
        # (op는 모두 0이므로 신경망 연산에 영향을 주지 않고 차원만 맞춰줍니다.)
        inputs = torch.cat([t, x, op], dim=1)

        u = self.model(inputs)

        total_physics_loss = 0.0

        for i in range(u.shape[1]):
            u_zone = u[:, i:i + 1]

            u_t = torch.autograd.grad(u_zone, t, grad_outputs=torch.ones_like(u_zone), create_graph=True)[0]
            u_x = torch.autograd.grad(u_zone, x, grad_outputs=torch.ones_like(u_zone), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

            # f = u_t - alpha * u_xx (열원이 없는 순수 열전도 방정식)
            f = u_t - self.salt_bath_alpha * u_xx
            total_physics_loss += torch.mean(f ** 2)

        return total_physics_loss / u.shape[1]