import threading
from Network.tcp_server import TCPServer

class Quenching(TCPServer):
    def __init__(self):
        TCPServer.__init__(self)

    def operate_quenching(self, materials=None):  # noqa
        print("퀜칭 동작")

        def server_task():
            # 🌟 포트 8085 및 안전장치 추가
            if self.start_server('127.0.0.1', 8085, timeout=30.0):
                self.send_data(materials)
                print(f"[Quenching] 컨베이어 벨트로 데이터 전송 완료")

                response = self.receive_data()
                if response:
                    print(f"[Quenching] 컨베이어 벨트로부터 응답 수신 완료: {response}")
            else:
                print("[!] Quenching: 컨베이어 벨트 접속 대기 타임아웃.")

            self.close()

        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def build_quenching(self):  # noqa
        print("퀜칭 설치")