import time
import pandas as pd
from Network.tcp_client import TCPClient


class ConveyorBelt(TCPClient):
    def __init__(self):
        TCPClient.__init__(self)

    def operate_conveyor_belt(self, materials=None):  # noqa
        print("컨베이어벨트 동작: 서버에 접속 시도")

        # 서버가 열릴 시간을 아주 잠깐 벌어줌
        time.sleep(0.5)

        received_materials = materials

        if self.connect_server('127.0.0.1', 8080):
            # 통신 부모 클래스가 압축을 풀어서 원본 DataFrame 객체로 돌려줌
            data = self.receive_data()

            if data is not None:
                received_materials = data

                if isinstance(received_materials, pd.DataFrame):
                    print(f"[ConveyorBelt] DataFrame 수신 완료! 크기: {received_materials.shape}")
                else:
                    print(f"[ConveyorBelt] 일반 데이터 수신 완료")

                time.sleep(1)  # 가상의 컨베이어 벨트 이송 시간

                # 피더로 작업 완료 상태 데이터 전송
                reply_message = {"status": "success", "message": "수신 및 이송 완료"}
                self.send_data(reply_message)
                print("[ConveyorBelt] 피더로 응답 전송 완료")

            self.close()
        else:
            print("[ConveyorBelt] 서버 연결 실패로 기존 인자값을 사용합니다.")

        return received_materials

    def build_conveyor_belt(self):  # noqa
        print("컨베이어벨트 생성")