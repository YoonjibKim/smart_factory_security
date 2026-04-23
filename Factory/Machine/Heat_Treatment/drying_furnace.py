import os
import torch
import threading
from Factory.PINN.pinn import PINN
from Network.tcp_server import TCPServer


class DryingFurnace(PINN, TCPServer):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)
        TCPServer.__init__(self)

        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path
        self.alpha = 0.01

        # 건조로 타겟 컬럼 설정
        self.__target_columns = [
            'TAG_MIN', '소입1존 OP', '소입2존 OP', '소입3존 OP', '소입4존 OP',
            '소입로 온도 1 Zone', '소입로 온도 2 Zone', '소입로 온도 3 Zone', '소입로 온도 4 Zone'
        ]
        self.__temp_cols = [
            '소입로 온도 1 Zone', '소입로 온도 2 Zone',
            '소입로 온도 3 Zone', '소입로 온도 4 Zone'
        ]
        self.__op_cols = [
            '소입1존 OP', '소입2존 OP', '소입3존 OP', '소입4존 OP'
        ]

    def get_target_columns(self):
        return self.__target_columns

    def operate_drying_furnace(self, materials=None):
        print("건조로 동작 및 추론(테스트) 수행")
        _, test_df = self._filter_features(test_df=materials)

        edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
        edge_model_path = os.path.join(edge_model_dir, "drying_furnace_hailo.onnx")

        # 엣지 모델 로드
        if os.path.exists(edge_model_path):
            self.__edge_session = self._load_model(edge_model_path)
            if self.__edge_session:
                print(f"[*] Edge 전용 두뇌(ONNX) 로드 완료: {edge_model_path}")
        else:
            print(f"[!] 경고: Edge 모델을 찾을 수 없습니다.")

        # PINN 추론 수행
        if test_df is not None:
            try:
                self._test_pinn(test_df, self.__temp_cols, self.__op_cols)
                self._simulate_what_if_op(test_df, self.__temp_cols, self.__op_cols, self.__op_cols[0])
                self._detect_anomalies(test_df, self.__temp_cols, self.__op_cols)
            except Exception as e:
                print(f"[!] PINN 추론 중 에러 발생 (건너뜜): {e}")

        # ==========================================
        # 네트워크 통신 (건조로 -> 컨베이어 벨트)
        # ==========================================
        def server_task():
            # 🌟 핵심 수정: 건조로는 8081 포트를 사용하며, 연결 성공 시에만 데이터를 보냅니다.
            # start_server가 True를 반환할 때까지(접속 시까지) 대기합니다.
            if self.start_server('127.0.0.1', 8081, timeout=30.0):
                # 클라이언트(컨베이어벨트)가 접속했을 때만 실행
                self.send_data(materials)
                print(f"[DryingFurnace] 컨베이어 벨트로 데이터 전송 완료")

                # 응답 확인
                response = self.receive_data()
                if response:
                    print(f"[DryingFurnace] 컨베이어 벨트로부터 응답 수신 완료: {response}")
            else:
                # 아무도 접속하지 않은 경우 에러 없이 로그만 출력
                print("[!] DryingFurnace: 컨베이어 벨트 접속 타임아웃.")

            self.close()

        # 통신 로직을 백그라운드 쓰레드에서 실행
        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def _filter_features(self, train_df=None, test_df=None):
        train_filtered = None
        test_filtered = None
        try:
            if train_df is not None:
                train_filtered = train_df[self.__target_columns].copy()
            if test_df is not None:
                test_filtered = test_df[self.__target_columns].copy()
            return train_filtered, test_filtered
        except KeyError as e:
            print(f"[!] 에러: 필요한 컬럼이 없습니다. {e}")
            return train_df, test_df

    def build_drying_furnace(self, train_dataset_path, test_dataset_path=None):
        print("건조로 설치 및 PINN 초기화")
        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, _ = self._filter_features(train_df=train_df, test_df=None)

        self._make_model()

        actual_model_path = os.path.join(self.pinn_model_path, "drying_furnace_pinn.pth") if os.path.isdir(
            self.pinn_model_path) else self.pinn_model_path

        if self.train_pinn_flag:
            print(f"[*] 학습 모드 작동: 건조로 모델 학습을 시작합니다.")
            self._train_pinn(filtered_train_df, self.__temp_cols, self.__op_cols, epochs=1000, sample_ratio=0.5)
            torch.save(self.model.state_dict(), actual_model_path)

            onnx_bytes = self._convert_onnx_model(self.model)
            if onnx_bytes:
                edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
                os.makedirs(edge_model_dir, exist_ok=True)
                edge_model_path = os.path.join(edge_model_dir, "drying_furnace_hailo.onnx")
                with open(edge_model_path, "wb") as f:
                    f.write(onnx_bytes)
        else:
            print(f"[*] 추론 모드 작동: 저장된 모델을 불러옵니다.")
            edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
            edge_model_path = os.path.join(edge_model_dir, "drying_furnace_hailo.onnx")

            if os.path.exists(edge_model_path):
                self.__edge_session = self._load_model(edge_model_path)
            elif os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path, map_location='cpu'))
                self.model.eval()
                onnx_bytes = self._convert_onnx_model(self.model)
                if onnx_bytes:
                    os.makedirs(edge_model_dir, exist_ok=True)
                    with open(edge_model_path, "wb") as f:
                        f.write(onnx_bytes)
                    self.__edge_session = self._load_model(edge_model_path)

    def _compute_physics_loss(self, t, x, op):
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = u_t - self.alpha * u_xx
        physics_loss = torch.mean(f ** 2)
        return physics_loss