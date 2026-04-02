import subprocess
import signal
import pandas as pd
import time
import os
import re

# Pandas 출력 옵션 설정 (콘솔에서 데이터프레임이 잘리지 않게 조정)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class PMU:
    def __init__(self, target_pct=80.0):
        # 현재 파이썬 프로세스 타겟팅 및 누적 점유율 임계값 설정
        self.target_pid = os.getpid()
        self.target_pct = target_pct
        self.stat_proc = None
        self.record_proc = None
        self.perf_data_file = f"perf_{self.target_pid}.data"

    def _start_perf_record(self):
        if self.record_proc: return
        print(f"[*] perf record 시작 준비 (PID: {self.target_pid}, 커널 스페이스 위주)...")
        # -F 99: 초당 99번 샘플링
        cmd = [
            "perf", "record", "-F", "99", "-e", "cycles:k",
            "-p", str(self.target_pid), "-o", self.perf_data_file
        ]
        self.record_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

    def _start_perf_stat(self):
        if self.stat_proc: return
        print(f"[*] perf stat 시작 준비 (PID: {self.target_pid}, 1초 간격)...")
        # -I 1000: 1000ms 간격 출력
        cmd = [
            "perf", "stat", "-I", "1000", "-x", ",",
            "-p", str(self.target_pid)
        ]
        self.stat_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid
        )

    def stop_and_collect(self):
        print("[*] 측정 종료 및 데이터 파싱 진행 중...")
        time.sleep(0.5)

        if self.stat_proc:
            try:
                os.killpg(os.getpgid(self.stat_proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

        if self.record_proc:
            try:
                os.killpg(os.getpgid(self.record_proc.pid), signal.SIGINT)
                self.record_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        # 1. stat 데이터 파싱
        _, stat_err = self.stat_proc.communicate()
        df_stat = self._parse_stat(stat_err)

        # 2. record/script 데이터 파싱
        df_top = pd.DataFrame()
        if os.path.exists(self.perf_data_file):
            print("[*] perf 스크립트 데이터 분석 진행 중...")
            script_result = subprocess.run(
                ["perf", "script", "-i", self.perf_data_file],
                capture_output=True, text=True
            )
            df_top = self._parse_script(script_result.stdout)
            os.remove(self.perf_data_file)
        else:
            print(f"[!] Warning: {self.perf_data_file} not found.")

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
        # stat 데이터는 기존 요청대로 reset_index() 유지
        return df.pivot_table(index='Time', columns='Event', values='Value').reset_index()

    def _parse_script(self, script_out):
        rows = []
        for line in script_out.splitlines():
            if "cycles:k" not in line: continue

            parts = line.split("cycles:k:")
            if len(parts) != 2: continue

            m_time = re.search(r'(\d+\.\d+):', parts[0])
            if not m_time: continue
            timestamp = float(m_time.group(1))

            sym_words = parts[1].strip().split()
            if not sym_words: continue

            if re.match(r'^[0-9a-f]+$', sym_words[0]) and len(sym_words) > 1:
                symbol_raw = " ".join(sym_words[1:])
            else:
                symbol_raw = " ".join(sym_words)

            symbol = re.sub(r'\+0x[0-9a-f]+', '', symbol_raw)
            symbol = re.sub(r'\s*\(\[.*\]\)', '', symbol).strip()

            rows.append({
                'Time_Raw': timestamp,
                'Symbol': f"[kernel] [k] {symbol}"
            })

        df = pd.DataFrame(rows)
        if df.empty: return df

        global_summary = df.groupby('Symbol').size().reset_index(name='Count')
        total_count = global_summary['Count'].sum()

        global_summary['Global_Overhead'] = (global_summary['Count'] / total_count * 100).round(2)
        global_summary = global_summary.sort_values('Global_Overhead', ascending=False)

        global_summary['Cum_Overhead'] = global_summary['Global_Overhead'].cumsum()
        top_symbols_df = global_summary[
            global_summary['Cum_Overhead'] - global_summary['Global_Overhead'] < self.target_pct].copy()

        target_symbols = top_symbols_df['Symbol'].tolist()

        min_time = df['Time_Raw'].min()
        df['Time_Idx'] = ((df['Time_Raw'] - min_time) // 1.0).astype(int) + 1

        total_per_sec = df.groupby('Time_Idx').size().reset_index(name='Total_Count')

        df_filtered = df[df['Symbol'].isin(target_symbols)]
        sec_summary = df_filtered.groupby(['Time_Idx', 'Symbol']).size().reset_index(name='Count')

        sec_summary = sec_summary.merge(total_per_sec, on='Time_Idx')
        sec_summary['Overhead'] = (sec_summary['Count'] / sec_summary['Total_Count'] * 100).round(2)

        # ---------------------------------------------------------
        # [핵심 수정] reset_index() 제거 -> Time_Idx가 고유 인덱스가 됨
        # ---------------------------------------------------------
        df_top_timeseries = sec_summary.pivot_table(
            index='Time_Idx',
            columns='Symbol',
            values='Overhead',
            fill_value=0.0
        )

        # 인덱스 이름표('Symbol') 제거하여 깔끔하게 정리
        df_top_timeseries.columns.name = None

        for sym in target_symbols:
            if sym not in df_top_timeseries.columns:
                df_top_timeseries[sym] = 0.0

        # DataFrame 컬럼 정렬 (Time_Idx는 이제 컬럼이 아니므로 타겟 심볼만 남김)
        df_top_timeseries = df_top_timeseries[target_symbols]

        return df_top_timeseries