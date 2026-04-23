import os
import torch
import threading
import pandas as pd
from Factory.PINN.pinn import PINN
from Network.tcp_server import TCPServer

class SaltBath(PINN, TCPServer):
    def __init__(self, train_pinn_flag, pinn_model_path):
        PINN.__init__(self)
        TCPServer.__init__(self)

        self.train_pinn_flag = train_pinn_flag
        self.pinn_model_path = pinn_model_path
        self.salt_bath_alpha = 0.01

        self.server_ip = '127.0.0.1'
        self.server_port = 8083

        self.__target_columns = [
            'TAG_MIN', '솔트조 온도 1 Zone', '솔트조 온도 2 Zone',
            '솔트_Dummy_OP_1', '솔트_Dummy_OP_2'
        ]
        self.__temp_cols = ['솔트조 온도 1 Zone', '솔트조 온도 2 Zone']
        self.__op_cols = ['솔트_Dummy_OP_1', '솔트_Dummy_OP_2']

    def get_target_columns(self):
        return self.__target_columns

    def operate_salt_bath(self, materials=None):  # noqa
        print("\n[SaltBath] 솔트조 동작 및 추론 수행")

        if isinstance(materials, pd.DataFrame):
            for col in self.__op_cols:
                if col not in materials.columns:
                    materials[col] = 0.0
                else:
                    materials[col] = 0.0 

        _, test_df = self._filter_features(test_df=materials)
        edge_model_dir = "Factory/Machine/Heat_Treatment/PINN/Edge_Model"
        edge_model_path = os.path.join(edge_model_dir, "salt_bath_hailo.onnx")

        if os.path.exists(edge_model_path):
            self.edge_session = self._load_model(edge_model_path)
            if self.edge_session and test_df is not None:
                self._test_pinn(test_df, self.__temp_cols, self.__op_cols)

        def server_task():
            print(f"[*] 가공 완료. 컨베이어벨트(Client)의 수거를 대기합니다... (서버 오픈: {self.server_ip}:{self.server_port})")
            try:
                if self.start_server(self.server_ip, self.server_port, timeout=30.0):
                    if self.conn:
                        # 🌟 핵심 수정: json 변환 없이 객체를 그대로 전송
                        self.send_data(materials)
                        response = self.receive_data()
                        print(f"[*] 컨베이어벨트 응답: {response}")
                else:
                    print("[!] SaltBath: 접속 대기 타임아웃.")
            except Exception as e:
                print(f"[!] 통신 에러 발생: {e}")
            finally:
                self.close()

        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def _filter_features(self, train_df=None, test_df=None):
        train_filtered, test_filtered = None, None
        try:
            if train_df is not None: train_filtered = train_df[self.__target_columns].copy()
            if test_df is not None: test_filtered = test_df[self.__target_columns].copy()
            return train_filtered, test_filtered
        except KeyError:
            return train_df, test_df

    def build_salt_bath(self, train_dataset_path, test_dataset_path=None):
        print("[*] 솔트조 설치 및 PINN 초기화 중...")
        train_df, test_df = self._load_dataset(train_dataset_path, test_dataset_path)
        filtered_train_df, _ = self._filter_features(train_df=train_df)
        self._make_model()

        actual_model_path = os.path.join(self.pinn_model_path, "salt_bath_pinn.pth") if os.path.isdir(self.pinn_model_path) else self.pinn_model_path

        if self.train_pinn_flag:
            self._train_pinn(train_df=filtered_train_df, target_cols=self.__temp_cols, op_cols=self.__op_cols, epochs=10000, sample_ratio=0.5)
            os.makedirs(os.path.dirname(actual_model_path), exist_ok=True)
            torch.save(self.model.state_dict(), actual_model_path)

            onnx_bytes = self._convert_onnx_model(self.model)
            if onnx_bytes:
                edge_model_path = "Factory/Machine/Heat_Treatment/PINN/Edge_Model/salt_bath_hailo.onnx"
                os.makedirs(os.path.dirname(edge_model_path), exist_ok=True)
                with open(edge_model_path, "wb") as f: f.write(onnx_bytes)
        else:
            edge_model_path = "Factory/Machine/Heat_Treatment/PINN/Edge_Model/salt_bath_hailo.onnx"
            if os.path.exists(edge_model_path):
                self.edge_session = self._load_model(edge_model_path)
            elif os.path.exists(actual_model_path):
                self.model.load_state_dict(torch.load(actual_model_path, map_location='cpu'))
                self.model.eval()
                onnx_bytes = self._convert_onnx_model(self.model)
                if onnx_bytes:
                    os.makedirs(os.path.dirname(edge_model_path), exist_ok=True)
                    with open(edge_model_path, "wb") as f: f.write(onnx_bytes)
                    self.edge_session = self._load_model(edge_model_path)

    def _compute_physics_loss(self, t, x, op):
        inputs = torch.cat([t, x, op], dim=1)
        u = self.model(inputs)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return torch.mean((u_t - self.salt_bath_alpha * u_xx) ** 2)