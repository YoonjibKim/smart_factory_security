import subprocess
import signal
import pandas as pd
import time
import os
import re

# Pandas 출력 제한 해제 (터미널에서 데이터 잘림 방지)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class PMU:
    def __init__(self, top_n=5):
        # 현재 실행 중인 파이썬 스크립트의 프로세스 ID를 자동으로 할당
        self.target_pid = os.getpid()
        self.top_n = top_n
        self.stat_proc = None
        self.record_proc = None
        # perf record 결과를 임시 저장할 파일명 지정
        self.perf_data_file = f"perf_{self.target_pid}.data"

    # 1. perf record 시작 (실시간 top 방식에서 백그라운드 record 방식으로 변경)
    def _start_perf_record(self):
        if self.record_proc: return
        print(f"[*] perf record 수집 시작 (PID: {self.target_pid}, 커널 오버헤드 전용)...")
        # -F 99: 초당 99번 샘플링 (1초 단위 빈도 분석에 최적화된 표준 수치)
        # -e cycles:k: 커널 영역(Kernel space)의 사이클만 측정
        # -o: 수집된 원본 데이터를 임시 파일로 저장
        cmd = [
            "perf", "record", "-F", "99", "-e", "cycles:k",
            "-p", str(self.target_pid), "-o", self.perf_data_file
        ]

        self.record_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )

    # 2. perf stat 시작 (1초 주기 동기화)
    def _start_perf_stat(self):
        if self.stat_proc: return
        print(f"[*] perf stat 수집 시작 (PID: {self.target_pid}, 1초 단위)...")
        # -I 1000: 1초마다 통계 출력
        cmd = [
            "perf", "stat", "-I", "1000", "-x", ",",
            "-p", str(self.target_pid)
        ]

        self.stat_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid
        )

    # 3. 측정 종료 및 데이터 파싱
    def stop_and_collect(self):
        print("[*] 측정 종료 및 데이터 병합 중...")

        # 마지막 샘플 확보를 위해 잠시 대기
        time.sleep(0.5)

        # 프로세스 종료 (perf record는 반드시 SIGINT로 종료해야 파일이 정상적으로 저장됨)
        if self.stat_proc:
            try:
                os.killpg(os.getpgid(self.stat_proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

        if self.record_proc:
            try:
                os.killpg(os.getpgid(self.record_proc.pid), signal.SIGINT)
                # perf.data 파일 작성이 완료될 때까지 대기
                self.record_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

        # perf stat 데이터 파싱
        _, stat_err = self.stat_proc.communicate()
        df_stat = self._parse_stat(stat_err)

        # perf script 데이터 파싱 (record 된 파일을 텍스트로 변환하여 분석)
        df_top = pd.DataFrame()
        if os.path.exists(self.perf_data_file):
            print("[*] perf 원본 데이터 시계열 변환 중...")
            script_result = subprocess.run(
                ["perf", "script", "-i", self.perf_data_file],
                capture_output=True, text=True
            )
            df_top = self._parse_script(script_result.stdout)

            # 변환이 끝난 임시 파일 삭제
            os.remove(self.perf_data_file)
        else:
            print(f"[!] Warning: {self.perf_data_file} not found.")

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

    def _parse_script(self, script_out):
        rows = []
        for line in script_out.splitlines():
            # 커널 사이클 이벤트만 필터링
            if "cycles:k" not in line: continue

            parts = line.split("cycles:k:")
            if len(parts) != 2: continue

            # 왼쪽 파트에서 타임스탬프 추출 (예: "python 16553 [000] 12345.101234:")
            m_time = re.search(r'(\d+\.\d+):', parts[0])
            if not m_time: continue
            timestamp = float(m_time.group(1))

            # 오른쪽 파트에서 주소 및 심볼 추출 (예: " ffffffff81123456 do_syscall_64+0x13 ([kernel.kallsyms])")
            sym_words = parts[1].strip().split()
            if not sym_words: continue

            # 첫 번째 단어가 16진수 주소면 제외하고 심볼명만 추출
            if re.match(r'^[0-9a-f]+$', sym_words[0]) and len(sym_words) > 1:
                symbol_raw = " ".join(sym_words[1:])
            else:
                symbol_raw = " ".join(sym_words)

            # 심볼명 뒤의 오프셋(+0x...) 및 모듈 정보([kernel.kallsyms]) 제거하여 깔끔하게 정리
            symbol = re.sub(r'\+0x[0-9a-f]+', '', symbol_raw)
            symbol = re.sub(r'\s*\(\[.*\]\)', '', symbol).strip()

            rows.append({
                'Time_Raw': timestamp,
                'Symbol': f"[kernel] [k] {symbol}"
            })

        df = pd.DataFrame(rows)
        if df.empty: return df

        # 타임스탬프를 1초 단위 인덱스로 변환 (perf stat의 1.0, 2.0 흐름과 동기화)
        min_time = df['Time_Raw'].min()
        df['Time_Idx'] = ((df['Time_Raw'] - min_time) // 1.0).astype(int) + 1

        # 1초 구간별로 심볼 등장 횟수(빈도수) 계산
        summary = df.groupby(['Time_Idx', 'Symbol']).size().reset_index(name='Count')

        # 빈도수를 기반으로 오버헤드 퍼센트(%) 계산
        total_per_sec = summary.groupby('Time_Idx')['Count'].transform('sum')
        summary['Overhead'] = (summary['Count'] / total_per_sec * 100).round(2)

        # 1초 구간별로 오버헤드가 높은 상위 top_n 개의 심볼만 추출
        final_df = summary.sort_values(['Time_Idx', 'Overhead'], ascending=[True, False]) \
            .groupby('Time_Idx').head(self.top_n).reset_index(drop=True)

        # 필요한 컬럼만 정리하여 반환
        return final_df[['Time_Idx', 'Overhead', 'Symbol']]