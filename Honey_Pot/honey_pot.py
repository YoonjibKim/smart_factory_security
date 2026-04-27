import os
import time
import csv

class HoneyPot:
    def __init__(self, log_dir="Honey_Pot/Result"):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "honeypot_trap_log.csv")
        os.makedirs(self.log_dir, exist_ok=True)

        # 🌟 기계가 바뀔 때마다 로그가 날아가지 않도록 'a' (이어쓰기) 모드로 유지
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Datetime', 'Target_Machine', 'Attack_Type', 'Sequence_Num', 'Details'])

        print(f"[Honeypot] 피지컬 AI 허니팟 시스템 대기 중... (로그: {self.log_file})")

    def trap_threat(self, machine_name, attack_type, timestamp, seq_num="None", details="None"):
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        with open(self.log_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, time_str, machine_name, attack_type, seq_num, details])