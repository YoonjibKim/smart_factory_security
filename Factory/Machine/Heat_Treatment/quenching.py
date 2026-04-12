import threading
from Network.tcp_server import TCPServer


class Quenching(TCPServer):
    def __init__(self):
        TCPServer.__init__(self)  # TCPServer 초기화

    def operate_quenching(self, materials=None):  # noqa
        print("퀜칭 동작")

        # ==========================================
        # 네트워크 통신 (퀜칭 -> 컨베이어 벨트)
        # ==========================================
        def server_task():
            # 다음 공정인 컨베이어 벨트가 접속할 수 있도록 서버 오픈 (포트 8080)
            self.start_server('127.0.0.1', 8080)

            # 피클(pickle) 기반으로 자동 압축되어 안전하게 전송됨
            self.send_data(materials)
            print(f"[Quenching] 컨베이어 벨트로 데이터 전송 완료")

            # 컨베이어 벨트가 보내는 완료 응답 수신 대기
            response = self.receive_data()
            if response:
                print(f"[Quenching] 컨베이어 벨트로부터 응답 수신 완료: {response}")

            self.close()

        # 메인 프로그램이 멈추지 않도록 백그라운드 쓰레드로 통신 실행
        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def build_quenching(self):  # noqa
        print("퀜칭 설치")