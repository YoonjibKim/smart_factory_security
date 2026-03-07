import multiprocessing as mp
import time
from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Matrials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials


# 🚨 클래스 외부에 독립적인 워커 함수 생성
def run_machine_process(process_name, target_method_name, preprocess_flag, train_flag, wake_up_event, done_event,
                        stop_event):
    machine = HeatTreatmentMachine(preprocess_flag, train_flag)
    target_method = getattr(machine, target_method_name)

    print(f"[*] {process_name} 프로세스 준비 완료. 대기(Sleep) 모드로 진입합니다...")

    while not stop_event.is_set():
        # 💤 메인 프로세스가 깨워줄 때까지 대기
        wake_up_event.wait()

        # 종료 신호가 켜져 있다면 루프 탈출
        if stop_event.is_set():
            break

        # 🚀 신호를 받으면 깨어나서 작업 수행
        print(f"\n>>> [{process_name}] 프로세스 깨어남(Wake-up)! 작업을 시작합니다.")
        target_method()
        print(f"[*] [{process_name}] 작업 완료. 다시 대기 모드로 돌아갑니다.")

        # 다음을 위해 wake_up_event를 끄고, 메인 프로세스에 끝났다고(done) 알림
        wake_up_event.clear()
        done_event.set()


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag, train_pinn_flag)

        self.preprocess_flag = preprocess_dataset_flag
        self.train_flag = train_pinn_flag

        self.processes = []
        # 프로세스별 개별 제어를 위한 이벤트 딕셔너리
        self.wake_up_events = {}
        self.done_events = {}
        self.stop_event = None

    def _operate_heat_treatment(self):  # noqa
        print("\n================ 열처리 공정 시작 ================")
        raw_part = self.__materials.load_pure_metal()
        self.__machine.operate_conveyor_belt(raw_part)

        # 1. drying_furnace (건조로) 깨우고 재우기
        self.wake_up_events["건조로"].set()
        self.done_events["건조로"].wait()
        self.done_events["건조로"].clear()
        self.__machine.operate_conveyor_belt()

        # 2. hardening_furnace (소입로) 깨우고 재우기
        self.wake_up_events["소입로"].set()
        self.done_events["소입로"].wait()
        self.done_events["소입로"].clear()
        self.__machine.operate_conveyor_belt()

        # 3. salt_bath (솔트조) 깨우고 재우기
        self.wake_up_events["솔트조"].set()
        self.done_events["솔트조"].wait()
        self.done_events["솔트조"].clear()
        self.__machine.operate_conveyor_belt()

        self.__machine.operate_quenching()
        self.__machine.operate_conveyor_belt()

        # 4. operate_washing (세정기) 깨우고 재우기
        self.wake_up_events["세정기"].set()
        self.done_events["세정기"].wait()
        self.done_events["세정기"].clear()
        self.__machine.operate_conveyor_belt()

        self.__machine.operate_tempering_furnace()
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_rust_prevention()
        self.__machine.operate_conveyor_belt()

        processed_part = self.__materials.unload_hardened_part()
        print("================ 열처리 공정 종료 ================\n")

        # 🚨 [수정됨] 공정이 모두 끝난 후 여기서 프로세스들을 깔끔하게 종료합니다.
        print("[!] 메인 프로세스: 모든 공정이 완료되었습니다. 자식 프로세스들을 종료합니다.")
        self.stop_event.set()

        # wait() 상태로 멈춰있는 자식 프로세스들의 잠을 깨워 루프를 탈출하게 함
        for event in self.wake_up_events.values():
            event.set()

        for p in self.processes:
            p.join()

        print("[*] 시스템이 메모리 누수 없이 안전하게 종료되었습니다.")

    def _build_heat_treatment_factory(self):
        print("열처리 공장 구축 (멀티프로세스 초기화 및 메모리 적재)")

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        ctx = mp.get_context('spawn')
        self.stop_event = ctx.Event()

        job_configs = [
            ("build_drying_furnace", "건조로"),
            ("build_hardening_furnace", "소입로"),
            ("build_salt_bath", "솔트조"),
            ("build_washing", "세정기")
        ]

        # 이벤트 객체 할당
        for _, process_name in job_configs:
            self.wake_up_events[process_name] = ctx.Event()
            self.done_events[process_name] = ctx.Event()

        # 4개의 자식 프로세스 생성 및 대기 상태 진입
        for method_name, process_name in job_configs:
            p = ctx.Process(
                target=run_machine_process,
                args=(
                    process_name, method_name, self.preprocess_flag, self.train_flag,
                    self.wake_up_events[process_name], self.done_events[process_name], self.stop_event
                )
            )
            p.daemon = True
            p.start()
            self.processes.append(p)

        print("\n[*] 4개의 공정 프로세스가 메모리에 적재되어 신호를 대기 중입니다.")
        time.sleep(1)  # 프로세스들이 준비될 때까지 잠깐 대기

        # 🚨 기존에 있던 self._operate_heat_treatment()와 프로세스 종료 로직을 제거했습니다!