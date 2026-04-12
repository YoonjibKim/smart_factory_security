import socket
import pickle
import struct


class TCPClient:
    def __init__(self):
        self._sock = None
        self._ip = None
        self._port = None

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None

    def connect_server(self, ip, port, timeout=60.0):
        self.close()
        self._ip = ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)

        try:
            self._sock.connect((self._ip, self._port))
            print(f"[Client] 서버({self._ip}:{self._port}) 접속 성공")
            return True
        except socket.timeout:
            print(f"[Client] 서버 접속 시간 초과({timeout}초).")
            self.close()
            return False
        except Exception as e:
            print(f"[Client] 서버 접속 실패: {e}")
            self.close()
            return False

    def _recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                print("[Client] 데이터 수신 시간 초과!")
                return None
            except Exception as e:
                print(f"[Client] 데이터 수신 에러: {e}")
                return None
        return bytes(data)

    def send_data(self, data):
        if self._sock:
            try:
                serialized = pickle.dumps(data)
                self._sock.sendall(struct.pack('>I', len(serialized)) + serialized)
            except Exception as e:
                print(f"[Client] 전송 에러: {e}")

    def receive_data(self):
        if self._sock:
            raw_msglen = self._recvall(self._sock, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]

            received_bytes = self._recvall(self._sock, msglen)
            if received_bytes:
                try:
                    return pickle.loads(received_bytes)
                except Exception as e:
                    print(f"[Client] 데이터 복원 에러: {e}")
                    return None
        return None