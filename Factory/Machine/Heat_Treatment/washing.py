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

        # 🚨 [수정됨] 업그레이드된 PINN.py는 동적 할당을 지원하므로
        # 더 이상 가짜 타겟(Dummy Zone 2)을 억지로 만들 필요가 없습니다!
        self._washing_temp_cols = [
            '세정기'
        ]

        # OP 데이터가 없는 설비이므로 타겟 개수(1개)에 맞춰 가짜 OP를 1개만 유지합니다.
        self._washing_op_cols = ['세정_Dummy_OP_1']

    def operate_washing(self, materials=None):  # noqa
        print("세정기 동작 및 추론(테스트) 수행")
        _, test_df = self._filter_features(test_df=materials)

        # 🚨 [핵심 버그 수정] 테스트 직전에 '세정기 전용 가중치(.pth)'를 무조건 다시 불러오기
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(
                self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "washing_pinn.pth")
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
            self._test_pinn(test_df, self._washing_temp_cols, self._washing_op_cols)
        else:
            print("[!] 경고: 입력된 원자재 데이터(raw_materials)가 없어 테스트를 건너뜁니다.")

        return materials

    def _filter_features(self, train_df=None, test_df=None):
        """
        세정기 PINN 학습에 필요한 핵심 센서 데이터를 추출합니다.
        (부모 클래스 규격에 맞춰 _filter_features로 이름 변경 및 None 처리 추가)
        """
        train_filtered = None
        test_filtered = None

        try:
            if train_df is not None:
                train_filtered = train_df[self._washing_target_columns].copy()
                # PINN.py 차원 맞춤용 가짜 OP 1개 추가
                train_filtered['세정_Dummy_OP_1'] = 0.0
                print(f"[*] 세정기 Train 데이터 필터링 완료: {train_filtered.columns.tolist()}")

            if test_df is not None:
                test_filtered = test_df[self._washing_target_columns].copy()
                test_filtered['세정_Dummy_OP_1'] = 0.0
                print(f"[*] 세정기 Test 데이터 필터링 완료: {test_filtered.columns.tolist()}")

            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 오타 확인 필요: {e}")
            return train_df, test_df

    def build_washing(self, train_dataset_path, test_dataset_path=None):
        print("세정기 설치 및 PINN 초기화")

        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, _ = self._filter_features(train_df=train_df, test_df=None)

        self._make_model()

        # 🚨 경로 자동 정리
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(
                self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "washing_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        # 🚨 학습 및 저장/로드 로직
        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 세정기 모델 학습을 시작합니다.")

            self._train_pinn(
                train_df=filtered_train_df,
                target_cols=self._washing_temp_cols,
                op_cols=self._washing_op_cols,
                epochs=10000,  # 다른 기계들과 동일하게 10000 에포크로 세팅
                sample_ratio=0.5
            )

            # 모델 저장 로직
            save_dir = os.path.dirname(actual_model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), actual_model_path)
            print(f"[*] 학습 완료: 모델이 성공적으로 저장되었습니다 -> {actual_model_path}")

        else:
            # 학습을 하지 않을 경우 저장된 모델 불러오기
            print(f"[*] 추론 모드 작동: 저장된 세정기 모델을 불러옵니다.")
            if os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path))
                self.model.eval()
                print(f"[*] 모델 로드 성공 -> {actual_model_path}")
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다 -> {actual_model_path}")

    def _compute_physics_loss(self, t, x, op):
        """
        세정기의 열전도 방정식 물리 손실 계산
        (수정된 pinn.py의 Stacked Input에 맞추어 1개 Zone만 깔끔하게 계산)
        """
        inputs = torch.cat([t, x, op], dim=1)

        # u의 shape: (batch_size, 1) -> 가짜 데이터를 없앴으므로 진짜 타겟 1개만 출력
        u = self.model(inputs)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # f = u_t - alpha * u_xx
        f = u_t - self.washing_alpha * u_xx

        physics_loss = torch.mean(f ** 2)
        return physics_loss