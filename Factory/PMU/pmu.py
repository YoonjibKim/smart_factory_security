import subprocess
import signal
import pandas as pd
import time
import os
import re


class PMU:
    def __init__(self, target_pid=None):
        self.target_pid = target_pid
        self.stat_proc = None
        self.top_proc = None

    # 1. perf top 시작 (sudo 제거, stdio 모드)
    def _start_perf_top(self):
        if self.top_proc: return
        print("[*] perf top 수집 시작...")
        # stdbuf: 라인 버퍼링 강제 (데이터 즉시 전달)
        # --stdio: 파이프라인 수집에 가장 최적화된 텍스트 모드
        cmd = ["stdbuf", "-oL", "-eL", "perf", "top", "--stdio", "-d", "1"]
        if self.target_pid:
            cmd.extend(["-p", str(self.target_pid)])

        self.top_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid
        )

    # 2. perf stat 시작 (100ms 주기)
    def _start_perf_stat(self):
        if self.stat_proc: return
        print("[*] perf stat 수집 시작 (100ms)...")
        cmd = ["perf", "stat", "-I", "100", "-x", ","]
        if self.target_pid:
            cmd.extend(["-p", str(self.target_pid)])
        else:
            cmd.append("-a")

        self.stat_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid
        )

    # 3. 데이터 파싱 및 종료
    def stop_and_collect(self):
        print("[*] 측정 종료 및 데이터 병합 중...")

        # 마지막 샘플 확보를 위해 잠시 대기
        time.sleep(0.5)

        # 프로세스 그룹 종료
        if self.stat_proc: os.killpg(os.getpgid(self.stat_proc.pid), signal.SIGINT)
        if self.top_proc: os.killpg(os.getpgid(self.top_proc.pid), signal.SIGTERM)

        stat_out, stat_err = self.stat_proc.communicate()
        top_out, _ = self.top_proc.communicate()

        df_stat = self._parse_stat(stat_err)
        df_top = self._parse_top(top_out)

        # 데이터가 하나라도 없으면 빈 DF 반환 방지
        if df_stat.empty: print("[!] Warning: Stat data is empty.")
        if df_top.empty: print("[!] Warning: Top data is empty.")

        return df_stat, df_top

    def _parse_stat(self, data_str):
        rows = []
        for line in data_str.splitlines():
            if not line.strip() or line.startswith('#'): continue
            p = line.split(',')
            if len(p) >= 4:
                try:
                    rows.append({'Time': float(p[0]), 'Event': p[3].strip(), 'Value': float(p[1])})
                except:
                    continue
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.pivot_table(index='Time', columns='Event', values='Value').reset_index()

    def _parse_top(self, data_str):
        rows = []
        current_time = 0
        # ANSI 제어 문자 제거 (stdio 모드에서도 가끔 섞임)
        clean_str = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', data_str)

        for line in clean_str.splitlines():
            line = line.strip()
            if any(h in line for h in ["Overhead", "Samples", "PerfTop:"]):
                current_time += 1
                continue

            p = line.split()
            if len(p) >= 3 and '%' in p[0]:
                try:
                    rows.append({
                        'Time_Idx': current_time,
                        'Overhead': float(p[0].replace('%', '')),
                        'Symbol': p[2]
                    })
                except:
                    continue

        df = pd.DataFrame(rows)
        if df.empty: return df
        # 매초 가장 점유율 높은 상위 1위 심볼만 추출
        return df.sort_values('Overhead', ascending=False).drop_duplicates('Time_Idx')