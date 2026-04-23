import threading
from Network.tcp_server import TCPServer

class Feeder(TCPServer):
    def __init__(self):
        TCPServer.__init__(self)  # 서버 역할 상속

    def operate_feeder(self, materials=None):
        print("피더 동작")

        # 네트워크 통신 (피더 -> 컨베이어 벨트)
        def server_task():
            # 다음 공정인 컨베이어 벨트가 접속할 수 있도록 서버 오픈 (포트 8080)
            self.start_server('127.0.0.1', 8080)

            # 데이터 전송
            self.send_data(materials)
            print(f"[Feeder] 컨베이어 벨트로 데이터 전송 완료")

            # 컨베이어 벨트의 응답 수신 대기
            response = self.receive_data()
            if response:
                print(f"[Feeder] 컨베이어 벨트로부터 응답 수신 완료: {response}")

            self.close()

        # 메인 프로그램이 멈추지 않도록 백그라운드 쓰레드로 실행
        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def build_feeder(self):
        print("피더 설치")