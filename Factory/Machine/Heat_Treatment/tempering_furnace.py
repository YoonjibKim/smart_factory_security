import pandas as pd
from Network.tcp_server import TCPServer


class TemperingFurnace(TCPServer):
    def __init__(self):
        # 상속 초기화
        TCPServer.__init__(self)

        # 네트워크 설정 (솔트조: 5000, 세정기: 5001, 소려로: 5002)
        self.server_ip = '127.0.0.1'  # 외부 통신 시 '0.0.0.0'
        self.server_port = 8080

    def operate_tempering_furnace(self, materials=None):  # noqa
        print("\n[TemperingFurnace] 소려로 동작 수행")

        # 추후 PINN 추론 로직이 추가될 경우 이 부분에 작성하시면 됩니다.

        # [네트워크 - 서버 대기 로직]
        print(f"[*] 가공 완료. 컨베이어벨트(Client)의 수거를 대기합니다... (서버 오픈: {self.server_ip}:{self.server_port})")
        try:
            self.start_server(self.server_ip, self.server_port)

            if self.conn:
                # DataFrame인 경우 JSON으로 직렬화, 그 외에는 문자열 변환 후 전송
                data_to_send = materials.to_json() if isinstance(materials, pd.DataFrame) else str(materials)
                self.send_data(data_to_send)

                # 수거 완료 응답 수신
                response = self.receive_data()
                print(f"[*] 컨베이어벨트 응답: {response}")

        except Exception as e:
            print(f"[!] 소려로 통신 에러 발생: {e}")
        finally:
            self.close()

        return materials

    def build_tempering_furnace(self):  # noqa
        print("소려로 설치")