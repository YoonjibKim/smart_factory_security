import os
import torch
from Factory.PINN.pinn import PINN


class DryingFurnace(PINN):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)

        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path

        self.alpha = 0.01

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

    def operate_drying_furnace(self, materials=None):
        print("건조로 동작 및 추론(테스트) 수행")
        _, test_df = self._filter_features(test_df=materials)

        # 🚨 [가장 중요한 핵심 버그 수정]
        # 다중 상속 공유 메모리 문제로 인해, 테스트 직전에 '자신의 진짜 가중치(.pth)'를 무조건 다시 불러와야 합니다.
        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "drying_furnace_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        if os.path.exists(actual_model_path):
            self.model.load_state_dict(torch.load(actual_model_path))
            self.model.eval()
            print(f"[*] 전용 두뇌 로드 완료: {actual_model_path}")
        else:
            print(f"[!] 경고: {actual_model_path} 모델을 찾을 수 없습니다.")

        if test_df is not None:
            print("[*] 원자재(Test 데이터)를 활용하여 모델 평가를 진행합니다.")
            self._test_pinn(test_df, self.__temp_cols, self.__op_cols)
        else:
            print("[!] 경고: 입력된 원자재 데이터(raw_materials)가 없어 테스트를 건너뜁니다.")

        return materials

    def _filter_features(self, train_df=None, test_df=None, target_columns=None):
        train_filtered = None
        test_filtered = None

        try:
            if train_df is not None:
                train_filtered = train_df[target_columns].copy()

            if test_df is not None:
                test_filtered = test_df[target_columns].copy()

            return train_filtered, test_filtered

        except KeyError as e:
            print(f"[!] 에러: CSV 파일에 필요한 컬럼이 없습니다. 오타 확인 필요: {e}")
            return train_df, test_df

    def build_drying_furnace(self, train_dataset_path, test_dataset_path=None):
        print("건조로 설치 및 PINN 초기화")

        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, _ = self._filter_features(train_df=train_df, test_df=None)

        self._make_model()

        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "drying_furnace_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 건조로 모델 학습을 시작합니다.")
            self._train_pinn(
                train_df=filtered_train_df,
                target_cols=self.__temp_cols,
                op_cols=self.__op_cols,
                epochs=10000,
                sample_ratio=0.5
            )

            save_dir = os.path.dirname(actual_model_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), actual_model_path)
            print(f"[*] 학습 완료: 모델이 성공적으로 저장되었습니다 -> {actual_model_path}")

        else:
            print(f"[*] 추론 모드 작동: 저장된 건조로 모델을 불러옵니다.")
            if os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path))
                self.model.eval()
                print(f"[*] 모델 로드 성공 -> {actual_model_path}")
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다 -> {actual_model_path}")

    def _compute_physics_loss(self, t, x, op):
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        f = u_t - self.alpha * u_xx
        physics_loss = torch.mean(f ** 2)
        return physics_loss