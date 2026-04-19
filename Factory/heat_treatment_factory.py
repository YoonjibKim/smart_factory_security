import time
from multiprocessing import Process, Queue, Event
from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Materials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials
from Factory.PMU.Model.attack_detection import AttackDetection


def detection_worker(data_queue, stop_event):
    """별도의 프로세스에서 무한 루프를 돌며 실시간으로 공격을 감시합니다."""
    detector = AttackDetection()  # 프로세스 독립적으로 모델 로드
    while not stop_event.is_set():
        if not data_queue.empty():
            task = data_queue.get()
            detector.detect(task['name'], task['data'])
        time.sleep(0.01)  # CPU 과점 방지


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag, train_pinn_flag)

        # 멀티 프로세스 통신을 위한 큐와 이벤트 설정
        self.data_queue = Queue()
        self.stop_event = Event()
        self.monitor_process = None

    def _extract_live_data(self, process_name, process_data):
        """가동 중인 머신의 데이터에서 모델 입력용 피처를 추출합니다."""
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
                    return process_data[available_cols].iloc[-1].tolist()
            return None
        except Exception:
            return None

    def _operate_heat_treatment(self):
        print("\n================ 열처리 공정 시작 (실시간 감시 가동) ================")

        # 🌟 공격 탐지 백그라운드 프로세스 시작
        self.monitor_process = Process(target=detection_worker, args=(self.data_queue, self.stop_event))
        self.monitor_process.start()

        try:
            # 1. 금속 재료 투입
            raw_materials = self.__materials.load_pure_metal()
            raw_materials = self.__machine.operate_feeder(raw_materials)
            raw_materials = self.__machine.operate_conveyor_belt(raw_materials)

            # 2. 건조로 공정
            drying_materials = self.__machine.operate_drying_furnace(raw_materials)
            # 🌟 데이터를 큐에 전송 (탐지 프로세스가 즉시 가져감)
            live_features = self._extract_live_data("건조로(Drying)", drying_materials)
            self.data_queue.put({'name': "건조로(Drying)", 'data': live_features})
            drying_materials = self.__machine.operate_conveyor_belt(drying_materials)

            # 3. 소입로(가열로) 공정
            hardening_materials = self.__machine.operate_hardening_furnace(drying_materials)
            live_features = self._extract_live_data("가열로(Hardening)", hardening_materials)
            self.data_queue.put({'name': "가열로(Hardening)", 'data': live_features})
            hardening_materials = self.__machine.operate_conveyor_belt(hardening_materials)

            # 4. 솔트조 공정
            salt_materials = self.__machine.operate_salt_bath(hardening_materials)
            live_features = self._extract_live_data("솔트배스(Salt Bath)", salt_materials)
            self.data_queue.put({'name': "솔트배스(Salt Bath)", 'data': live_features})
            salt_materials = self.__machine.operate_conveyor_belt(salt_materials)

            # 5. 퀜칭 공정
            quenching_materials = self.__machine.operate_quenching(salt_materials)
            quenching_materials = self.__machine.operate_conveyor_belt(quenching_materials)

            # 6. 세정기 공정
            washing_materials = self.__machine.operate_washing(quenching_materials)
            live_features = self._extract_live_data("세척기(Washing)", washing_materials)
            self.data_queue.put({'name': "세척기(Washing)", 'data': live_features})
            washing_materials = self.__machine.operate_conveyor_belt(washing_materials)

            # 9. 부품 가공 완료
            processed_part = self.__materials.unload_hardened_part(washing_materials)
            print("================ 열처리 공정 종료 ================\n")

        finally:
            # 공정 종료 후 감시 프로세스 안전하게 종료
            self.stop_event.set()
            self.monitor_process.join()