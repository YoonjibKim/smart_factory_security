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

    def _operate_hardening_furnace(self):  # noqa
        print("소입로 동작")

    def _filter_features(self, train_df, test_df):
        """
        전체 데이터셋에서 '소입로' PINN 학습에 필요한 핵심 센서 데이터만 추출합니다.
        """
        try:
            train_filtered = train_df[self.__target_columns].copy()
            test_filtered = test_df[self.__target_columns].copy()

            print(f"[*] 소입로 데이터 필터링 완료: {train_filtered.columns.tolist()}")
            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 확인 필요: {e}")
            return train_df, test_df

    def _build_hardening_furnace(self, train_dataset_path, test_dataset_path):  # noqa
        print("소입로 설치 및 PINN 초기화")

        # 1. 데이터 로드 및 필터링
        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, filtered_test_df = self._filter_features(train_df, test_df)

        # 2. 모델 초기화
        self._make_model()

        # 🚨 경로 자동 정리: 넘어온 경로가 폴더일 경우 파일명 강제 지정
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(
                self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "hardening_furnace_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        # 🚨 train_pinn_flag에 따른 조건부 학습 및 저장/로드 로직
        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 소입로 모델 학습을 시작합니다.")

            # 3. 모델 학습
            self._train_pinn(
                train_df=filtered_train_df,
                target_cols=self.__temp_cols,
                op_cols=self.__op_cols,
                epochs=10000,
                sample_ratio=0.5
            )

            # 4. 모델 테스트 (평가)
            self._test_pinn(filtered_test_df, self.__temp_cols, self.__op_cols)

            # 5. 모델 저장 로직
            save_dir = os.path.dirname(actual_model_path)
            if save_dir:  # 경로에 폴더가 포함되어 있다면 폴더 생성
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), actual_model_path)
            print(f"[*] 학습 완료: 모델이 성공적으로 저장되었습니다 -> {actual_model_path}")

        else:
            # 학습을 하지 않을 경우 저장된 모델 불러오기
            print(f"[*] 추론 모드 작동: 저장된 소입로 모델을 불러옵니다.")
            if os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path))
                self.model.eval()  # 평가 모드로 전환
                print(f"[*] 모델 로드 성공 -> {actual_model_path}")
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다 -> {actual_model_path}")

    def _compute_physics_loss(self, t, x, op):
        """
        소입로의 1차원 열전도 방정식 물리 손실 계산 (4개 Zone 개별 계산)
        """
        # 신경망의 입력: 시간(t), 위치(x), 제어값(op)
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)  # u의 shape: (batch_size, 4) -> 1~4존 온도

        total_physics_loss = 0.0

        # 소입로 1존 ~ 4존까지 각각 편미분 및 열방정식 계산
        for i in range(u.shape[1]):
            u_zone = u[:, i:i + 1]  # 각 Zone의 온도만 추출 (batch_size, 1)

            # 시간(t)에 대한 1차 편미분 (du/dt)
            u_t = torch.autograd.grad(
                u_zone, t,
                grad_outputs=torch.ones_like(u_zone),
                create_graph=True
            )[0]

            # 공간(x)에 대한 1차 편미분 (du/dx)
            u_x = torch.autograd.grad(
                u_zone, x,
                grad_outputs=torch.ones_like(u_zone),
                create_graph=True
            )[0]

            # 공간(x)에 대한 2차 편미분 (d^2u/dx^2)
            u_xx = torch.autograd.grad(
                u_x, x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True
            )[0]

            # f = du/dt - alpha * (d^2u/dx^2)
            # (추후 고도화 시 여기에 각 존별 OP(화력)에 의한 열원 추가 및 복사열 공식 추가 가능)
            f = u_t - self.alpha * u_xx

            # 각 존의 물리 손실을 누적
            total_physics_loss += torch.mean(f ** 2)

        # 4개 존의 평균 물리 손실 반환
        return total_physics_loss / u.shape[1]