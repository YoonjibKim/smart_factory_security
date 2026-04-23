import time
import pandas as pd
from Network.tcp_client import TCPClient


class ConveyorBelt(TCPClient):
    def __init__(self):
        TCPClient.__init__(self)

    def operate_conveyor_belt(self, materials=None, target_port=8080):
        print(f"컨베이어벨트 동작: 서버(포트:{target_port})에 접속 시도")
        received_materials = materials

        connected = False
        # 최대 10번(5초간) 재시도하여 Connection Refused 방지
        for i in range(10):
            time.sleep(0.5)
            if self.connect_server('127.0.0.1', target_port):
                connected = True
                break
            print(f"   -> 서버가 열리길 기다리는 중... ({i + 1}/10)")

        if connected:
            data = self.receive_data()
            if data is not None:
                received_materials = data
                if isinstance(received_materials, pd.DataFrame):
                    print(f"[ConveyorBelt] DataFrame 수신 완료! 크기: {received_materials.shape}")
                else:
                    print(f"[ConveyorBelt] 일반 데이터 수신 완료")

                time.sleep(1)

                # 피더로 작업 완료 상태 데이터 전송
                reply_message = {"status": "success", "message": "수신 및 이송 완료"}
                self.send_data(reply_message)
                print("[ConveyorBelt] 피더로 응답 전송 완료")

            # 🌟 핵심 수정: 부모의 close() 함수를 찾지 못하는 에러를 방지하기 위해
            # 소켓 연결(conn)이 살아있다면 직접 통신을 종료하도록 변경했습니다.
            if self.conn:
                self.conn.close()
                self.conn = None

        else:
            print("[ConveyorBelt] 서버 연결 최종 실패. 기존 데이터를 그대로 반환합니다.")

        return received_materials

    def build_conveyor_belt(self):
        print("컨베이어벨트 생성")