import os
import torch
from Factory.PINN.pinn import PINN


class HardeningFurnace(PINN):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)

        # 🚨 파라미터로 받은 변수를 클래스 내부에서 사용할 수 있도록 저장
        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path

        # 소입로 열확산 계수 (건조로와 동일하게 임의 설정, 실제 공정에 맞게 튜닝 필요)
        self.alpha = 0.01

        # 1. 전체 데이터 추출용 리스트 (소입로는 1~4존까지 존재)
        self.__target_columns = [
            'TAG_MIN',  # 시간 변수 (t를 만들기 위한 원본 데이터)
            '소입1존 OP',  # 1존 제어/출력값 (입력 특성)
            '소입2존 OP',  # 2존 제어/출력값 (입력 특성)
            '소입3존 OP',  # 3존 제어/출력값 (입력 특성)
            '소입4존 OP',  # 4존 제어/출력값 (입력 특성)
            '소입로 온도 1 Zone',  # 1존 온도 (예측 타겟 u)
            '소입로 온도 2 Zone',  # 2존 온도 (예측 타겟 u)
            '소입로 온도 3 Zone',  # 3존 온도 (예측 타겟 u)
            '소입로 온도 4 Zone'  # 4존 온도 (예측 타겟 u)
        ]

        # 2. PINN 공간 맵핑을 위한 역할 분담 리스트
        self.__temp_cols = [
            '소입로 온도 1 Zone',
            '소입로 온도 2 Zone',
            '소입로 온도 3 Zone',
            '소입로 온도 4 Zone'
        ]
        self.__op_cols = [
            '소입1존 OP',
            '소입2존 OP',
            '소입3존 OP',
            '소입4존 OP'
        ]

    def operate_hardening_furnace(self, materials=None):
        print("소입로 동작 및 추론(테스트) 수행")
        _, test_df = self._filter_features(test_df=materials)

        # 🚨 [핵심 버그 수정] 테스트 직전에 '소입로 전용 가중치(.pth)'를 무조건 다시 불러오기
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "hardening_furnace_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        if os.path.exists(actual_model_path):
            self.model.load_state_dict(torch.load(actual_model_path))
            self.model.eval()
            print(f"[*] 전용 두뇌 로드 완료: {actual_model_path}")
        else:
            print(f"[!] 경고: {actual_model_path} 모델을 찾을 수 없습니다.")

        # 테스트 수행
        if test_df is not None:
            print("[*] 원자재(Test 데이터)를 활용하여 모델 평가를 진행합니다.")
            self._test_pinn(test_df, self.__temp_cols, self.__op_cols)
            self._simulate_what_if_op(test_df, self.__temp_cols, self.__op_cols, self.__op_cols[0])
            self._detect_anomalies(test_df, self.__temp_cols, self.__op_cols)
        else:
            print("[!] 경고: 입력된 원자재 데이터(raw_materials)가 없어 테스트를 건너뜁니다.")

        return materials

    def _filter_features(self, train_df=None, test_df=None):
        """
        전체 데이터셋에서 '소입로' PINN 학습에 필요한 핵심 센서 데이터만 추출합니다.
        """
        train_filtered = None
        test_filtered = None

        try:
            if train_df is not None:
                train_filtered = train_df[self.__target_columns].copy()
                print(f"[*] 소입로 Train 데이터 필터링 완료: {train_filtered.columns.tolist()}")

            if test_df is not None:
                test_filtered = test_df[self.__target_columns].copy()
                print(f"[*] 소입로 Test 데이터 필터링 완료: {test_filtered.columns.tolist()}")

            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 확인 필요: {e}")
            return train_df, test_df


    def build_hardening_furnace(self, train_dataset_path, test_dataset_path=None):
        print("소입로 설치 및 PINN 초기화")

        # 1. 데이터 로드 및 필터링
        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, _ = self._filter_features(train_df=train_df, test_df=None)

        # 2. 모델 초기화
        self._make_model()

        # 🚨 경로 자동 정리
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "hardening_furnace_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        # 🚨 학습 및 저장/로드 로직
        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 소입로 모델 학습을 시작합니다.")

            # 3. 모델 학습
            self._train_pinn(
                train_df=filtered_train_df,
                target_cols=self.__temp_cols,
                op_cols=self.__op_cols,
                epochs=10000,  # 다른 기계들과 동일하게 10000 에포크로 세팅
                sample_ratio=0.5
            )

            # 4. 모델 저장 로직
            save_dir = os.path.dirname(actual_model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), actual_model_path)
            print(f"[*] 학습 완료: 모델이 성공적으로 저장되었습니다 -> {actual_model_path}")

        else:
            # 학습을 하지 않을 경우 저장된 모델 불러오기
            print(f"[*] 추론 모드 작동: 저장된 소입로 모델을 불러옵니다.")
            if os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path))
                self.model.eval()
                print(f"[*] 모델 로드 성공 -> {actual_model_path}")
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다 -> {actual_model_path}")


    def _compute_physics_loss(self, t, x, op):
        """
        소입로의 1차원 열전도 방정식 물리 손실 계산
        (수정된 pinn.py의 Stacked Input에 맞추어 통합 편미분 계산으로 최적화)
        """
        # 신경망의 입력은 3개 (시간 t, 위치 x, 제어값 op)
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)

        # 물리 법칙(편미분)은 시간(t)과 공간(x)에 대해서만 계산
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # f = du/dt - alpha * (d^2u/dx^2)
        f = u_t - self.alpha * u_xx

        physics_loss = torch.mean(f ** 2)
        return physics_loss