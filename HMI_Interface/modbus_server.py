from pyModbusTCP.server import ModbusServer


class ModbusTCPServer:
    def __init__(self, ip='0.0.0.0', port=502):
        self._ip = ip
        self._port = port
        # no_block=True로 설정하면 백그라운드 스레드에서 서버가 비동기로 돌아갑니다.
        self.server = ModbusServer(host=self._ip, port=self._port, no_block=True)

    def start_server(self):
        print(f"[{self._port}] 포트에서 Modbus TCP 서버 대기 중...")
        self.server.start()

    def stop_server(self):
        self.server.stop()
        print("서버가 안전하게 종료되었습니다.")

    # --- 기존의 send_data/receive_data 대신 레지스터 값을 직접 조작합니다 ---

    def set_data(self, address, values):
        """
        서버 내부의 Holding Register 값을 업데이트합니다.
        address: 시작 주소 (예: 0)
        values: 저장할 정수 리스트 (예: [10, 20, 30])
        """
        if isinstance(values, int):
            values = [values]
        self.server.data_bank.set_holding_registers(address, values)
        # print(f"[서버 내부] 주소 {address}에 값 {values} 저장됨")

    def get_data(self, address, word_length=1):
        """
        서버 내부의 Holding Register 값을 읽어옵니다.
        """
        return self.server.data_bank.get_holding_registers(address, word_length)