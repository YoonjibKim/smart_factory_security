import threading
from Network.tcp_server import TCPServer


class Feeder(TCPServer):
    def __init__(self):
        TCPServer.__init__(self)

    def operate_feeder(self, materials=None):  # noqa
        print("공급기 동작: 서버 가동")

        def server_task():
            self.start_server('127.0.0.1', 8080)

            # DataFrame 등 파이썬 객체를 그대로 넘기면 부모 클래스에서 알아서 압축 변환 후 전송
            self.send_data(materials)
            print(f"[Feeder] 컨베이어 벨트로 데이터 전송 완료")

            # 컨베이어 벨트가 보내는 완료 응답 대기
            response = self.receive_data()
            if response:
                print(f"[Feeder] 컨베이어 벨트로부터 응답 수신 완료: {response}")

            self.close()

        thread = threading.Thread(target=server_task)
        thread.daemon = True
        thread.start()

        return materials

    def build_feeder(self):  # noqa
        print("공급기 설치")