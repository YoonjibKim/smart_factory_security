import socket
import pickle
import struct


class TCPServer:
    def __init__(self):
        self._sock = None
        self.conn = None
        self._ip = None
        self._port = None

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None
        if self._sock:
            try:
                self._sock.close()
            except:
                pass
            self._sock = None

    def start_server(self, ip, port, timeout=60.0):
        self.close()
        self._ip = ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._ip, self._port))
        self._sock.listen(5)

        self._sock.settimeout(timeout)
        print(f"[Server] {self._port} 포트에서 연결 대기 중... (최대 {timeout}초 대기)")

        try:
            self.conn, self.addr = self._sock.accept()
            self.conn.settimeout(timeout)
            print(f"[Server] 클라이언트 접속 완료: {self.addr}")
        except socket.timeout:
            print(f"[Server] {timeout}초 동안 접속이 없어 대기를 종료합니다.")
            self.close()
        except Exception as e:
            print(f"[Server] 연결 대기 중 에러: {e}")
            self.close()

    def _recvall(self, sock, n):
        """정해진 바이트 크기(n)만큼 정확히 데이터를 읽어오는 도우미 함수"""
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.timeout:
                print("[Server] 데이터 수신 시간 초과!")
                return None
            except Exception as e:
                print(f"[Server] 데이터 수신 에러: {e}")
                return None
        return bytes(data)

    def send_data(self, data):
        if self.conn:
            try:
                # 데이터를 압축된 바이너리로 변환 (아주 빠름)
                serialized = pickle.dumps(data)
                # 데이터의 총 길이(4바이트 정수)를 본문 앞에 붙여서 한 번에 전송
                self.conn.sendall(struct.pack('>I', len(serialized)) + serialized)
            except Exception as e:
                print(f"[Server] 전송 에러: {e}")

    def receive_data(self):
        if self.conn:
            # 1. 처음 4바이트를 읽어 앞으로 올 본문 데이터의 총 길이를 파악
            raw_msglen = self._recvall(self.conn, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]

            # 2. 파악한 길이만큼만 정확하게 본문 데이터 수신
            received_bytes = self._recvall(self.conn, msglen)
            if received_bytes:
                try:
                    # 바이너리를 다시 파이썬 객체(DataFrame 등)로 즉시 복원
                    return pickle.loads(received_bytes)
                except Exception as e:
                    print(f"[Server] 데이터 복원 에러: {e}")
                    return None
        return None