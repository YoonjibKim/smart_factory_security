import torch
from Factory.PINN.pinn import PINN


class DryingFurnace(PINN):
    def __init__(self):
        PINN.__init__(self)
        self.alpha = 0.01

        # 🚨 건조로 컬럼이 아닌 '소입로' 컬럼으로 정확히 지정되어야 합니다.
        self.__target_columns = [
            'TAG_MIN',
            '소입1존 OP',
            '소입2존 OP',
            '소입3존 OP',
            '소입4존 OP',
            '소입로 온도 1 Zone',
            '소입로 온도 2 Zone',
            '소입로 온도 3 Zone',
            '소입로 온도 4 Zone'
        ]

        self.__temp_cols = [
            '소입로 온도 1 Zone', '소입로 온도 2 Zone',
            '소입로 온도 3 Zone', '소입로 온도 4 Zone'
        ]
        self.__op_cols = [
            '소입1존 OP', '소입2존 OP',
            '소입3존 OP', '소입4존 OP'
        ]

    def _operate_drying_furnace(self):  # noqa
        print("건조로 동작")

    def _filter_features(self, train_df, test_df):
        """
        '소입로' 데이터만 추출하는 필터링 메서드
        """
        try:
            train_filtered = train_df[self.__target_columns].copy()
            test_filtered = test_df[self.__target_columns].copy()

            # 프린트문도 소입로로 변경하여 로그에서 제대로 확인 가능하도록 수정
            print(f"[*] 소입로 데이터 필터링 완료: {train_filtered.columns.tolist()}")
            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 오타 확인 필요: {e}")
            return train_df, test_df

    def _build_drying_furnace(self, train_dataset_path, test_dataset_path):
        print("건조로 설치 및 PINN 초기화")

        # 1. 데이터 로드 및 필터링
        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, filtered_test_df = self._filter_features(train_df, test_df)

        # 2. 모델 초기화
        self._make_model()

        # 3. 모델 학습 (파라미터로 온도와 OP 리스트를 분리해서 넘김!)
        self._train_pinn(
            train_df=filtered_train_df,
            target_cols=self.__temp_cols,
            op_cols=self.__op_cols,
            epochs=10000,
            sample_ratio=0.5
        )

        # 4. 모델 테스트 (평가)
        self._test_pinn(filtered_test_df, self.__temp_cols, self.__op_cols)

    def _compute_physics_loss(self, t, x, op):  # 🚨 op 파라미터 추가
        """
        건조로의 1차원 열전도 방정식(Heat Equation) 물리 손실 계산
        """
        # 신경망의 입력은 3개 (시간 t, 위치 x, 제어값 op)
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)

        # 물리 법칙(편미분)은 시간(t)과 공간(x)에 대해서만 계산
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # f = du/dt - alpha * (d^2u/dx^2)
        # (OP는 온도를 올리는 외부 열원이므로 나중에 f 수식에 + 화력(OP) 요소로 추가할 수도 있음!)
        f = u_t - self.alpha * u_xx

        physics_loss = torch.mean(f ** 2)
        return physics_loss