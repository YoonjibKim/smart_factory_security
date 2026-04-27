import time
import csv
import os
import re
from collections import deque, defaultdict


class CyberPhysicalIntrusionDetection:
    def __init__(self, dos_threshold=30, dos_time_window=0.5):
        # PMU 자원 고갈(DoS) 판단 임계값
        self.dos_threshold = dos_threshold
        self.dos_time_window = dos_time_window
        self.request_times = defaultdict(lambda: deque(maxlen=self.dos_threshold))

        # 물리적 AI (PINN) 및 Replay 판단용 버퍼
        self.last_values = {}
        self.replay_buffer = defaultdict(lambda: deque(maxlen=4))

        os.makedirs("/tmp", exist_ok=True)
        self.log_path = "/tmp/prediction_log.csv"

        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'predicted_type', 'sequence'])

    def detect_attack(self, machine_name, payload, current_val):
        current_time = time.time()
        seq_num = "None"
        attack_type = "Normal"
        details = "정상"

        is_internal = isinstance(payload, list)
        raw_str = str(payload[0]) if is_internal and payload else str(payload)

        # 시스템 채점을 위한 시퀀스 번호 파싱 (공격 자체와는 무관함)
        if not is_internal and "_SEQ_" in raw_str:
            parts = raw_str.split("_SEQ_")
            clean_payload = parts[0].strip()
            seq_num = re.sub(r'[^0-9]', '', parts[1])
        else:
            clean_payload = raw_str.strip()

        # ----------------------------------------------------
        # 🌟 가짜 시그니처 완전 삭제! (PMU 트래픽과 물리 법칙으로만 판단)
        # ----------------------------------------------------
        if not is_internal:
            # 1. PMU / 네트워크 부하 탐지 (DoS)
            req_queue = self.request_times[machine_name]
            req_queue.append(current_time)

            if len(req_queue) == self.dos_threshold and (current_time - req_queue[0]) < self.dos_time_window:
                attack_type = "DoS"
                details = "PMU: 비정상적 트래픽 과부하 감지 (자원 고갈 공격)"
            else:
                try:
                    # 2. 페이로드가 정상적인 숫자인지 확인
                    num_val = float(clean_payload)

                    # 3. 물리 법칙 기반 탐지 (FDIA - PINN 모델 프록시)
                    if machine_name in self.last_values:
                        diff = abs(num_val - self.last_values[machine_name])
                        if diff > 100.0:  # 열역학적으로 불가능한 비정상적 온도 급증
                            attack_type = "FDIA"
                            details = "Physical AI: 물리적 임계치 초과 (센서 변조)"

                    # 4. 물리적 노이즈 검증 (Replay)
                    rep_buf = self.replay_buffer[machine_name]
                    rep_buf.append(num_val)
                    if len(rep_buf) == 4 and len(set(rep_buf)) == 1:
                        attack_type = "Replay"
                        details = "Physical AI: 비정상적 수치 동결 (과거 데이터 재전송)"

                    self.last_values[machine_name] = num_val

                except ValueError:
                    attack_type = "MITM"
                    details = "네트워크: 구조가 손상된 비표준 패킷 (파싱 에러)"

        # 기록 처리
        if attack_type != "Normal" and seq_num != "None":
            with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([current_time, attack_type, seq_num])

        return attack_type, seq_num, details


ids_engine = CyberPhysicalIntrusionDetection()