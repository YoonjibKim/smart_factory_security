import time
import os
import threading
import pandas as pd
from multiprocessing import Process, Queue, Event
from sklearn.metrics import accuracy_score, f1_score

from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Materials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials
from Factory.PMU.Model.attack_detection import ids_engine
from Honey_Pot.honey_pot import HoneyPot


def detection_worker(data_queue, stop_event):
    honeypot = HoneyPot(log_dir="Honey_Pot/Result")
    while not stop_event.is_set() or not data_queue.empty():
        if not data_queue.empty():
            try:
                task = data_queue.get_nowait()
                attack_type, seq_num, details = ids_engine.detect_attack(task['name'], task['data'],
                                                                         task['data'][0] if task['data'] else 0)
                if attack_type != "Normal":
                    honeypot.trap_threat(task['name'], attack_type, time.time(), seq_num=seq_num, details=details)
            except Exception:
                pass
        else:
            time.sleep(0.01)


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag, train_pinn_flag)
        self.data_queue = Queue()
        self.stop_event = Event()
        self.monitor_process = None

    def _modbus_attack_monitor(self, machine_name, register_addr, stop_event):
        honeypot = HoneyPot(log_dir="Honey_Pot/Result")
        print(f"   🛡️ [{machine_name}] 외부 공격 감시망 가동 (주소: {register_addr})")

        while not stop_event.is_set():
            try:
                words = self.__machine.get_server_state(register_addr, 30)
                if words and any(w != 0 for w in words):
                    chars = [chr(w >> 8) + chr(w & 0xFF) for w in words if w != 0]
                    payload = "".join(chars).replace('\x00', '')

                    if payload and "Normal" not in payload:
                        attack_type, seq_num, details = ids_engine.detect_attack(machine_name, payload, 0)
                        if attack_type != "Normal":
                            honeypot.trap_threat(machine_name, attack_type, time.time(), seq_num=seq_num,
                                                 details=details)
                        self.__machine.clear_server_state(register_addr, 30)
            except Exception:
                pass
            time.sleep(0.005)

    def _stream_live_data(self, process_name, process_data):
        try:
            if type(process_data).__name__ == 'DataFrame':
                for _, row in process_data.head(300).iterrows():
                    self.data_queue.put({'name': process_name, 'data': row.tolist()})
        except:
            pass

    def _update_server_state(self, status_code):
        for _ in range(3):
            try:
                self.__machine.set_server_state(999, status_code)
                time.sleep(0.05)
            except:
                pass

    def _execute_machine(self, machine_name, address, machine_func, *args, **kwargs):
        monitor_stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self._modbus_attack_monitor,
                                          args=(machine_name, address, monitor_stop_event))
        monitor_thread.daemon = True
        monitor_thread.start()

        self._update_server_state(address)
        time.sleep(0.2)

        result = machine_func(*args, **kwargs)
        self._stream_live_data(machine_name, result)

        time.sleep(2.5)

        self._update_server_state(0)
        time.sleep(0.2)

        monitor_stop_event.set()
        monitor_thread.join()
        return result

    def _evaluate_performance(self):
        print("\n📊 공정 종료. 물리적 AI 방어 시스템의 최종 성적표를 계산합니다...\n")
        try:
            # 🌟 HMI가 생성한 정답지 위치를 공용 폴더인 /tmp/ground_truth.csv로 고정!
            gt_path = '/tmp/ground_truth.csv'
            pred_path = '/tmp/prediction_log.csv'

            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                print("⚠️ 채점용 데이터 파일이 부족하여 성적표를 출력할 수 없습니다.")
                return

            gt_df = pd.read_csv(gt_path)

            try:
                with open(pred_path, 'r') as f:
                    first_line = f.readline()
                if 'sequence' in first_line:
                    pred_df = pd.read_csv(pred_path)
                else:
                    pred_df = pd.read_csv(pred_path, names=['timestamp', 'predicted_type', 'sequence'])
            except pd.errors.EmptyDataError:
                pred_df = pd.DataFrame(columns=['timestamp', 'predicted_type', 'sequence'])

            gt_df['sequence'] = gt_df['sequence'].astype(str).str.strip()
            pred_df['sequence'] = pred_df['sequence'].astype(str).str.strip()

            pred_df = pred_df.drop_duplicates(subset=['sequence'], keep='last')
            merged_df = pd.merge(gt_df, pred_df, on='sequence', how='left')

            y_true = merged_df['attack_type']
            y_pred = merged_df['predicted_type'].fillna('Lost')

            total_sent = len(gt_df)
            total_received = (y_pred != 'Lost').sum()

            print("==================================================")
            print("        🏆 물리적 AI 방어 시스템 성능 🏆")
            print("==================================================")
            print(f" ▶ 총 유입 패킷: {total_sent} 개")
            print(f" ▶ 방어 성공   : {total_received} 개")
            print(
                f" ▶ 현실 유실률 : {((total_sent - total_received) / total_sent * 100) if total_sent else 0:.2f}% (네트워크 Delay)")
            print("-" * 50)

            if not y_true.empty:
                atype = y_true.unique()[0]
                y_t = (y_true == atype).astype(int)
                y_p = (y_pred == atype).astype(int)

                acc = accuracy_score(y_t, y_p)
                f1 = f1_score(y_t, y_p, zero_division=0)

                print(f" 🎯 [{atype}] 실질 탐지 정확도 (수신된 패킷 기준)")
                print(f"    - Accuracy : {acc * 100:>6.2f}%")
                print(f"    - F1-Score : {f1 * 100:>6.2f}%")
            print("==================================================\n")
        except Exception as e:
            print(f"⚠️ 채점 중 오류: {e}")

    def _operate_heat_treatment(self):
        print("\n[INFO] 피지컬 AI 방어 시스템 (독립 병렬 방어 모드) 가동 중...")

        # 🌟 서버 시작 시 임시 파일 완벽 초기화
        for log_file in ['/tmp/prediction_log.csv', '/tmp/ground_truth.csv', 'Honey_Pot/Result/honeypot_trap_log.csv']:
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except:
                    pass

        self.monitor_process = Process(target=detection_worker, args=(self.data_queue, self.stop_event))
        self.monitor_process.start()
        self._update_server_state(0)

        try:
            raw = self.__materials.load_pure_metal()
            raw = self.__machine.operate_feeder(raw)
            raw = self.__machine.operate_conveyor_belt(raw, port=8080)

            drying = self._execute_machine("건조로(Drying)", 120, self.__machine.operate_drying_furnace, raw)
            drying = self.__machine.operate_conveyor_belt(drying, port=8081)

            hardening = self._execute_machine("가열로(Hardening)", 130, self.__machine.operate_hardening_furnace, drying)
            hardening = self.__machine.operate_conveyor_belt(hardening, port=8082)

            salt = self._execute_machine("솔트배스(Salt Bath)", 150, self.__machine.operate_salt_bath, hardening)
            salt = self.__machine.operate_conveyor_belt(salt, port=8083)

            quenching = self.__machine.operate_quenching(salt)
            quenching = self.__machine.operate_conveyor_belt(quenching, port=8085)

            washing = self._execute_machine("세척기(Washing)", 110, self.__machine.operate_washing, quenching)
            washing = self.__machine.operate_conveyor_belt(washing, port=8084)

            processed = self.__materials.unload_hardened_part(washing)

            self._update_server_state(9999)
            print("\n[INFO] 모든 공정이 안전하게 완료되었습니다.")

        finally:
            self.stop_event.set()
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except:
                    break
            if hasattr(self.data_queue, 'cancel_join_thread'):
                self.data_queue.cancel_join_thread()
            if self.monitor_process:
                self.monitor_process.join(timeout=3)

            time.sleep(1)
            self._evaluate_performance()