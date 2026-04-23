import pandas as pd
import threading
from Network.tcp_server import TCPServer

class RustPrevention(TCPServer):
    def __init__(self):
        TCPServer.__init__(self)
        self.server_ip = '127.0.0.1'
        self.server_port = 8086

    def operate_rust_prevention(self, materials=None):  # noqa
        print("\n[RustPrevention] 방청기 동작 수행")

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
                    print("[!] RustPrevention: 접속 대기 타임아웃.")
            except Exception as e:
                print(f"[!] 방청기 통신 에러 발생: {e}")
            finally:
                self.close()

        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def build_rust_prevention(self):  # noqa
        print("방청기 설치")