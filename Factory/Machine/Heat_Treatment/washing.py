import os
import torch
from Factory.PINN.pinn import PINN


class Washing(PINN):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)

        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path

        self.washing_alpha = 0.01

        self.__target_columns = [
            'TAG_MIN',
            '세정기'
        ]

        self.__temp_cols = [
            '세정기'
        ]

        self.__op_cols = ['세정_Dummy_OP_1']

    def operate_washing(self, materials=None):  # noqa
        print("세정기 동작 및 추론(테스트) 수행 (Edge 모델 전용)")
        _, test_df = self._filter_features(test_df=materials)

        edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
        edge_model_path = os.path.join(edge_model_dir, "washing_hailo.onnx")

        # 무조건 엣지 모델(ONNX)만 로드
        if os.path.exists(edge_model_path):
            self.edge_session = self._load_model(edge_model_path)
            if self.edge_session:
                print(f"[*] Edge 전용 두뇌(ONNX) 로드 완료: {edge_model_path}")
        else:
            raise FileNotFoundError(f"[!] 에러: Edge 모델을 찾을 수 없습니다. 빌드(build)를 먼저 수행하여 ONNX 모델을 생성하세요. 경로: {edge_model_path}")

        if test_df is not None:
            print("[*] 원자재(Test 데이터)를 활용하여 Edge 모델 평가를 진행합니다.")
            self._test_pinn(test_df, self.__temp_cols, self.__op_cols)
            self._simulate_what_if_op(test_df, self.__temp_cols, self.__op_cols, self.__op_cols[0])
            self._detect_anomalies(test_df, self.__temp_cols, self.__op_cols)
        else:
            print("[!] 경고: 입력된 원자재 데이터(raw_materials)가 없어 테스트를 건너뜁니다.")

        return materials

    def _filter_features(self, train_df=None, test_df=None):
        train_filtered = None
        test_filtered = None

        try:
            if train_df is not None:
                train_filtered = train_df[self.__target_columns].copy()
                train_filtered['세정_Dummy_OP_1'] = 0.0
                print(f"[*] 세정기 Train 데이터 필터링 완료: {train_filtered.columns.tolist()}")

            if test_df is not None:
                test_filtered = test_df[self.__target_columns].copy()
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

        if self.pinn_model_path.endswith('/') or self.pinn_model_path.endswith('\\') or os.path.isdir(
                self.pinn_model_path):
            actual_model_path = os.path.join(self.pinn_model_path, "washing_pinn.pth")
        else:
            actual_model_path = self.pinn_model_path

        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 세정기 모델 학습을 시작합니다.")

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

            print("[*] Edge 모델(ONNX) 변환을 시작합니다...")
            onnx_bytes = self._convert_onnx_model(self.model)

            if onnx_bytes:
                edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
                os.makedirs(edge_model_dir, exist_ok=True)

                edge_model_path = os.path.join(edge_model_dir, "washing_hailo.onnx")

                with open(edge_model_path, "wb") as f:
                    f.write(onnx_bytes)

                print(f"[*] 변환된 Edge 모델(ONNX) 저장 완료 -> {edge_model_path}")

        else:
            print(f"[*] 추론 모드 작동: 저장된 모델을 불러옵니다.")

            edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
            edge_model_path = os.path.join(edge_model_dir, "washing_hailo.onnx")

            if os.path.exists(edge_model_path):
                print(f"[*] Edge 모델 발견! 로드를 시도합니다 -> {edge_model_path}")
                self.edge_session = self._load_model(edge_model_path)
                if self.edge_session:
                    print("[*] Edge 모델(ONNX) 로드 성공!")

            elif os.path.exists(actual_model_path):
                print(f"[*] Edge 모델이 없습니다. 기존 PyTorch 모델을 로드하여 즉시 ONNX로 변환합니다 -> {actual_model_path}")

                self.model.load_state_dict(torch.load(actual_model_path, map_location='cpu'))
                self.model.eval()
                print(f"[*] PyTorch 모델 로드 성공! (CPU 메모리 안착)")

                print("[*] 기존 모델을 바탕으로 Edge 모델(ONNX) 자동 변환을 시작합니다...")
                onnx_bytes = self._convert_onnx_model(self.model)

                if onnx_bytes:
                    os.makedirs(edge_model_dir, exist_ok=True)
                    with open(edge_model_path, "wb") as f:
                        f.write(onnx_bytes)
                    print(f"[*] ✅ 자동 변환 및 Edge 모델(ONNX) 저장 완료 -> {edge_model_path}")

                    self.edge_session = self._load_model(edge_model_path)
            else:
                print(f"[!] 에러: 지정된 경로에서 모델을 찾을 수 없습니다.")

    def _compute_physics_loss(self, t, x, op):
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        f = u_t - self.washing_alpha * u_xx
        physics_loss = torch.mean(f ** 2)
        return physics_loss