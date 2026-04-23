import socket
import pickle
import struct


class TCPClient:
    def __init__(self):
        self.conn = None

    def connect_server(self, host, port, timeout=10.0):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.settimeout(timeout)
        try:
            self.conn.connect((host, port))
            return True
        except Exception:
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
            # 서버가 보낸 8바이트 헤더(길이) 먼저 읽기
            raw_length = self._recvall(self.conn, 8)
            if not raw_length:
                return None
            msg_length = struct.unpack('>Q', raw_length)[0]

            # 길이만큼 데이터 조립
            data = self._recvall(self.conn, msg_length)
            if data:
                return pickle.loads(data)
        except Exception as e:
            print(f"[!] Client 수신 에러: {e}")
        return None