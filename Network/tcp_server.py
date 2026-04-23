import socket
import pickle
import struct


class TCPServer:
    def __init__(self):
        self.server_socket = None
        self.conn = None
        self.client_address = None

    def start_server(self, host, port, timeout=15.0):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # TIME_WAIT 무시 (포트 즉시 재사용 가능하게 설정)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((host, port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(timeout)

            self.conn, self.client_address = self.server_socket.accept()
            return True
        except Exception as e:
            print(f"[!] Server 연결 에러 (포트 {port}): {e}")
            return False

    # tcp_server.py 발췌
    def send_data(self, data):
        serialized = pickle.dumps(data)
        # 데이터 길이를 8바이트 정수로 변환하여 헤더로 붙임
        length_prefix = struct.pack('>Q', len(serialized))
        self.conn.sendall(length_prefix + serialized)

    def _recvall(self, conn, n):
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet: return None
            data.extend(packet)
        return bytes(data)

    def receive_data(self):
        if not self.conn:
            return None
        try:
            # 1. 맨 앞 8바이트(길이 정보) 먼저 수신
            raw_length = self._recvall(self.conn, 8)
            if not raw_length:
                return None
            msg_length = struct.unpack('>Q', raw_length)[0]

            # 2. 약속된 길이만큼 정확히 본문 수신
            data = self._recvall(self.conn, msg_length)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"[!] Server 수신 에러: {e}")
        return None

    def close(self):
        if self.conn:
            self.conn.close()
        if self.server_socket:
            self.server_socket.close()