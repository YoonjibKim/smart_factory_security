import socket
import pickle
import struct
import time

class TCPServer:
    def __init__(self):
        self._sock = None
        self.conn = None  # 기존 공정 코드(if self.conn:) 호환성을 위한 더미 플래그
        self.clients = [] # [(conn, addr), ...] 다중 클라이언트 소켓 배열
        self._ip = None
        self._port = None

    def close(self):
        # 연결된 모든 클라이언트 세션 종료
        for c, addr in self.clients:
            try: c.close()
            except: pass
        self.clients = []
        self.conn = None
        if self._sock:
            try: self._sock.close()
            except: pass
        self._sock = None

    def start_server(self, ip, port, wait_time=2.0):
        self.close()
        self._ip = ip
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._ip, self._port))
        self._sock.listen(5)

        print(f"[Server] {self._port} 포트 오픈. {wait_time}초 동안 다중 접속(정상+공격) 대기 중...")

        self._sock.settimeout(0.5) # 0.5초 단위로 끊어서 다수의 접속을 확인
        start_time = time.time()

        # 주어진 시간(wait_time) 동안 접속하는 모든 클라이언트를 받아들임
        while time.time() - start_time < wait_time:
            try:
                c, addr = self._sock.accept()
                c.settimeout(5.0)
                print(f"[Server] 🚨 새로운 클라이언트 접속 포착: {addr}")
                self.clients.append((c, addr))
            except socket.timeout:
                continue
            except Exception as e:
                break

        if self.clients:
            self.conn = True  # 자식 클래스(공정)의 if self.conn: 패스를 위한 플래그
        else:
            print(f"[Server] 접속한 클라이언트가 없습니다.")

    def send_data(self, data):
        if not self.clients: return
        serialized = pickle.dumps(data)
        packet = struct.pack('>I', len(serialized)) + serialized

        # 연결된 모든 클라이언트(컨베이어벨트, 해커)에게 데이터 브로드캐스팅
        for c, addr in self.clients:
            try:
                c.sendall(packet)
                print(f"[Server] -> {addr} 로 데이터 전송 완료")
            except Exception as e:
                print(f"[Server] -> {addr} 전송 에러: {e}")

    def receive_data(self):
        responses = []
        for c, addr in self.clients:
            try:
                raw_msglen = self._recvall(c, 4)
                if not raw_msglen: continue
                msglen = struct.unpack('>I', raw_msglen)[0]
                received_bytes = self._recvall(c, msglen)
                if received_bytes:
                    resp = pickle.loads(received_bytes)
                    responses.append({f"{addr[0]}:{addr[1]}": resp})
            except Exception as e:
                pass
        return responses

    def _recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet: return None
                data.extend(packet)
            except socket.timeout:
                return None
            except Exception:
                return None
        return bytes(data)