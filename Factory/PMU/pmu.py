import subprocess
import signal
import pandas as pd


class PMU:
    def __init__(self, target_pid=None):
        """
        :param target_pid: 모니터링할 특정 프로세스 ID (None일 경우 시스템 전체 모니터링)
        """
        self.target_pid = target_pid
        self.stat_process = None
        self.top_process = None

    # ==========================================
    # 1. perf top (1초 단위 함수 점유율 캡처)
    # ==========================================
    def _start_perf_top(self):
        if self.top_process is not None:
            print("[-] perf top이 이미 실행 중입니다.")
            return

        print("[*] perf top 측정 시작 (1초 단위, 메모리 기록)...")
        # -b: 배치 모드 (화면 클리어 없이 쭉 출력)
        # -d 1: 1초 단위 갱신
        cmd = ["perf", "top", "-b", "-d", "1"]
        if self.target_pid:
            cmd.extend(["-p", str(self.target_pid)])

        # stdout으로 출력되는 배치 데이터를 메모리에 캡처
        self.top_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def _end_perf_top(self):
        if self.top_process is None:
            print("[-] 실행 중인 perf top이 없습니다.")
            return pd.DataFrame()

        print("[*] perf top 측정 종료 및 DataFrame 생성 중...")
        # perf top은 SIGTERM으로 종료
        self.top_process.send_signal(signal.SIGTERM)

        stdout_data, _ = self.top_process.communicate()
        self.top_process = None

        df = self.__parse_top_string_to_df(stdout_data)
        print("[+] perf top DataFrame 변환 완료!")
        return df

    def __parse_top_string_to_df(self, output_str):
        """perf top의 배치 출력을 파싱하여 1초 단위 DataFrame으로 변환"""
        data = []
        current_time_sec = 0

        for line in output_str.splitlines():
            line = line.strip()
            # 새로운 스냅샷(1초 경과)이 시작될 때마다 시간을 1씩 증가
            if "PerfTop:" in line or "Events:" in line:
                current_time_sec += 1
                continue

            # 파싱 예시: "12.34%  [kernel]  [k] _raw_spin_unlock_irqrestore"
            parts = line.split(maxsplit=2)
            if len(parts) >= 3 and parts[0].endswith('%'):
                try:
                    overhead = float(parts[0].replace('%', ''))
                    shared_obj = parts[1]
                    symbol = parts[2]

                    data.append({
                        'Time (s)': current_time_sec,
                        'Overhead (%)': overhead,
                        'Shared Object': shared_obj,
                        'Symbol': symbol
                    })
                except ValueError:
                    continue

        return pd.DataFrame(data)

    # ==========================================
    # 2. perf stat (1초 단위 하드웨어/소프트웨어 이벤트 캡처)
    # ==========================================
    def _start_perf_stat(self):
        if self.stat_process is not None:
            print("[-] perf stat이 이미 실행 중입니다.")
            return

        print("[*] perf stat 측정 시작 (1초 단위, 메모리 기록)...")
        cmd = ["perf", "stat", "-I", "1000", "-x", ","]
        if self.target_pid:
            cmd.extend(["-p", str(self.target_pid)])
        else:
            cmd.append("-a")

        # stderr로 출력되는 CSV 데이터를 메모리에 캡처
        self.stat_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def _end_perf_stat(self):
        if self.stat_process is None:
            print("[-] 실행 중인 perf stat이 없습니다.")
            return pd.DataFrame()

        print("[*] perf stat 측정 종료 및 DataFrame 생성 중...")
        self.stat_process.send_signal(signal.SIGINT)

        _, stderr_data = self.stat_process.communicate()
        self.stat_process = None

        df = self.__parse_stat_string_to_df(stderr_data)
        print("[+] perf stat DataFrame 변환 완료!")
        return df

    def __parse_stat_string_to_df(self, output_str): # noqa
        """perf stat의 CSV 문자열을 파싱하여 DataFrame으로 변환"""
        data = []
        for line in output_str.splitlines():
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    time_val = float(parts[0])
                    value_str = parts[1].strip()
                    value = float(value_str) if value_str and value_str[0].isdigit() else None
                    event_name = parts[3].strip()

                    data.append({
                        'Time (s)': time_val,
                        'Event': event_name,
                        'Value': value
                    })
                except ValueError:
                    continue

        raw_df = pd.DataFrame(data)
        if raw_df.empty:
            return raw_df

        # 시간을 인덱스로, 각 이벤트를 컬럼으로 피벗
        df_pivot = raw_df.pivot_table(index='Time (s)', columns='Event', values='Value').reset_index()
        return df_pivot