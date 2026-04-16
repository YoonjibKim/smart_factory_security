import atexit  # 프로그램 종료 시 자동 실행을 위한 모듈 추가
from HMI_Interface.modbus_server import ModbusTCPServer


class HMIinterface(ModbusTCPServer):
    def __init__(self, ip='0.0.0.0', port=5020):  # 포트 기본값을 5020으로 변경
        # 부모 클래스(ModbusTCPServer) 초기화
        super().__init__(ip, port)

        # 1. 객체 선언과 동시에 자동으로 서버 시작!
        self.start_server()

        # 2. 메인 프로그램이 종료될 때 자동으로 stop_server()가 실행되도록 예약!
        atexit.register(self.stop_server)

    def send_message(self, address, message):
        """
        기계(서버)에서 HMI로 보낼 데이터를 레지스터에 기록합니다.
        """
        if isinstance(message, str):
            encoded_bytes = message.encode('utf-8')
            if len(encoded_bytes) % 2 != 0:
                encoded_bytes += b'\x00'

            values = []
            for i in range(0, len(encoded_bytes), 2):
                val = (encoded_bytes[i] << 8) | encoded_bytes[i + 1]
                values.append(val)
            self.set_data(address, values)
        else:
            self.set_data(address, message)

    def receive_message(self, address, length=1, is_string=False):
        """
        HMI가 기계(서버)의 레지스터에 쓴 데이터를 읽어옵니다.
        """
        data = self.get_data(address, length)

        if data and is_string:
            byte_array = bytearray()
            for val in data:
                byte_array.append((val >> 8) & 0xFF)
                byte_array.append(val & 0xFF)

            return byte_array.decode('utf-8', errors='ignore').rstrip('\x00')

        return data