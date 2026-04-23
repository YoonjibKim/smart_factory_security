import time
from multiprocessing import Process, Queue, Event
from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Materials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials
from Factory.PMU.Model.attack_detection import PMUStateMonitor


def detection_worker(data_queue, stop_event):
    """별도의 프로세스에서 무한 루프를 돌며 실시간으로 공정 상태를 감시합니다."""
    detector = PMUStateMonitor()  # 프로세스 독립적으로 모델 로드
    while not stop_event.is_set():
        if not data_queue.empty():
            task = data_queue.get()
            detector.detect(task['name'], task['data'])
        else:
            time.sleep(0.01)  # CPU 과점 방지 및 데이터 대기


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag, train_pinn_flag)

        # 멀티 프로세스 통신을 위한 큐와 이벤트 설정
        self.data_queue = Queue()
        self.stop_event = Event()
        self.monitor_process = None

    def _stream_live_data(self, process_name, process_data):
        """
        가동된 머신의 전체 시계열 데이터(DataFrame)를 받아
        한 줄씩(Row-by-Row) 큐에 넣어 실시간 PMU 센서 스트리밍을 모사합니다.
        """
        machine_type = None
        if "Drying" in process_name:
            machine_type = self.__machine.MachineType.DRYING
        elif "Hardening" in process_name:
            machine_type = self.__machine.MachineType.HARDENING
        elif "Salt Bath" in process_name:
            machine_type = self.__machine.MachineType.SALT_BATH
        elif "Washing" in process_name:
            machine_type = self.__machine.MachineType.WASHING

        try:
            if type(process_data).__name__ == 'DataFrame':
                if machine_type is not None:
                    target_cols = self.__machine.get_target_columns(machine_type=machine_type)
                    available_cols = [col for col in target_cols if col in process_data.columns]

                    # 🌟 핵심: 58만 개의 데이터를 Queue에 한 번에 넣으면 OS가 멈춥니다.
                    # PMU 테스트용으로 앞의 200~500개 행만 추출하여 스트리밍합니다.
                    for _, row in process_data[available_cols].head(200).iterrows():
                        self.data_queue.put({'name': process_name, 'data': row.tolist()})

        except Exception as e:
            print(f"데이터 스트리밍 중 오류 발생: {e}")

    def _operate_heat_treatment(self):
        print("\n================ 열처리 공정 시작 (PMU 상태 실시간 감시 가동) ================")

        # 상태 감시 백그라운드 프로세스 시작
        self.monitor_process = Process(target=detection_worker, args=(self.data_queue, self.stop_event))
        self.monitor_process.start()

        try:
            # 1. 금속 재료 투입
            raw_materials = self.__materials.load_pure_metal()
            raw_materials = self.__machine.operate_feeder(raw_materials)

            # 🌟 [포트 수정] 피더 - 포트 8080
            raw_materials = self.__machine.operate_conveyor_belt(raw_materials, port=8080)

            # 2. 건조로 공정
            drying_materials = self.__machine.operate_drying_furnace(raw_materials)
            self._stream_live_data("건조로(Drying)", drying_materials)

            # 🌟 [포트 수정] 건조로 - 포트 8081
            drying_materials = self.__machine.operate_conveyor_belt(drying_materials, port=8081)

            # 3. 소입로(가열로) 공정
            hardening_materials = self.__machine.operate_hardening_furnace(drying_materials)
            self._stream_live_data("가열로(Hardening)", hardening_materials)

            # 🌟 [포트 수정] 소입로 - 포트 8082
            hardening_materials = self.__machine.operate_conveyor_belt(hardening_materials, port=8082)

            # 4. 솔트조 공정
            salt_materials = self.__machine.operate_salt_bath(hardening_materials)
            self._stream_live_data("솔트배스(Salt Bath)", salt_materials)

            # 🌟 [포트 수정] 솔트배스 - 포트 8083
            salt_materials = self.__machine.operate_conveyor_belt(salt_materials, port=8083)

            # 5. 퀜칭 공정
            quenching_materials = self.__machine.operate_quenching(salt_materials)

            # 🌟 [포트 수정] 퀜칭 - 포트 8085
            quenching_materials = self.__machine.operate_conveyor_belt(quenching_materials, port=8085)

            # 6. 세정기 공정
            washing_materials = self.__machine.operate_washing(quenching_materials)
            self._stream_live_data("세척기(Washing)", washing_materials)

            # 🌟 [포트 수정] 세척기 - 포트 8084
            washing_materials = self.__machine.operate_conveyor_belt(washing_materials, port=8084)

            # 9. 부품 가공 완료
            processed_part = self.__materials.unload_hardened_part(washing_materials)
            print("================ 열처리 공정 종료 ================\n")

        finally:
            # 공정 종료 후 큐에 쌓인 남은 데이터들이 탐지 모델에서 처리될 시간을 약간 부여
            time.sleep(1)
            # 감시 프로세스 안전하게 종료
            self.stop_event.set()
            self.monitor_process.join()