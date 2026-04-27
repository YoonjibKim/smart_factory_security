from Factory.Machine.Heat_Treatment.conveyor_belt import ConveyorBelt
from Factory.Machine.Heat_Treatment.drying_furnace import DryingFurnace
from Factory.Machine.Heat_Treatment.feeder import Feeder
from Factory.Machine.Heat_Treatment.hardening_furnace import HardeningFurnace
from Factory.Machine.Heat_Treatment.quenching import Quenching
from Factory.Machine.Heat_Treatment.rust_prevention import RustPrevention
from Factory.Machine.Heat_Treatment.salt_bath import SaltBath
from Factory.Machine.Heat_Treatment.tempering_furnace import TemperingFurnace
from Factory.Machine.Heat_Treatment.washing import Washing
from Factory.PMU.pmu import PMU
from enum import Enum
from pyModbusTCP.server import ModbusServer, DataBank
import pandas as pd
import os


class HeatTreatmentMachine(PMU):
    class MachineType(Enum):
        DRYING = 1
        HARDENING = 2
        SALT_BATH = 3
        WASHING = 4

    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        PMU.__init__(self)

        # 🌟 클라이언트 연결이 아닌 Modbus 서버 직접 구동
        self.server = ModbusServer(host='0.0.0.0', port=5020, no_block=True)
        try:
            self.server.start()
            print("[5020] 포트에서 Modbus TCP 서버 구동 완료...")
        except Exception as e:
            print(f"[!] Modbus 서버 시작 실패: {e}")

        raw_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/품질전처리후데이터.csv"
        quality_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/열처리_품질데이터.xlsx"
        self.__train_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Train/train_dataset.csv"
        self.__test_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Test/test_dataset.csv"

        self.__feeder = Feeder()
        self.__conveyor_belt = ConveyorBelt()
        self.__drying_furnace = DryingFurnace(train_pinn_flag,
                                              "Factory/Machine/Heat_Treatment/PINN/Model/drying_furnace_model.pth")
        self.__hardening_furnace = HardeningFurnace(train_pinn_flag,
                                                    "Factory/Machine/Heat_Treatment/PINN/Model/hardening_furnace_model.pth")
        self.__salt_bath = SaltBath(train_pinn_flag, "Factory/Machine/Heat_Treatment/PINN/Model/salt_bath_model.pth")
        self.__quenching = Quenching()
        self.__washing = Washing(train_pinn_flag, "Factory/Machine/Heat_Treatment/PINN/Model/washing_model.pth")
        self.__tempering_furnace = TemperingFurnace()
        self.__rust_prevention = RustPrevention()

        if preprocess_dataset_flag:
            self.__preprocess_data(raw_dataset_path, quality_dataset_path)

    # ----------------------------------------------------
    # 🌟 Modbus 서버 메모리 읽기/쓰기/지우기 기능 지원
    # ----------------------------------------------------
    def set_server_state(self, address, value):
        if hasattr(self.server, 'data_bank'):
            self.server.data_bank.set_holding_registers(address, [value])
        else:
            DataBank.set_words(address, [value])

    def get_server_state(self, address, count=30):
        if hasattr(self.server, 'data_bank'):
            res = self.server.data_bank.get_holding_registers(address, count)
            return res if res else [0] * count
        else:
            res = DataBank.get_words(address, count)
            return res if res else [0] * count

    def clear_server_state(self, address, count=30):
        if hasattr(self.server, 'data_bank'):
            self.server.data_bank.set_holding_registers(address, [0] * count)
        else:
            DataBank.set_words(address, [0] * count)

    def send_message(self, address, message):
        if isinstance(message, str):
            encoded_bytes = message.encode('utf-8')
            if len(encoded_bytes) % 2 != 0: encoded_bytes += b'\x00'
            values = [(encoded_bytes[i] << 8) | encoded_bytes[i + 1] for i in range(0, len(encoded_bytes), 2)]
            if hasattr(self.server, 'data_bank'):
                self.server.data_bank.set_holding_registers(address, values)
            else:
                DataBank.set_words(address, values)
        else:
            val = [message] if isinstance(message, int) else message
            if hasattr(self.server, 'data_bank'):
                self.server.data_bank.set_holding_registers(address, val)
            else:
                DataBank.set_words(address, val)

    def _extract_safe_value(self, data):
        if isinstance(data, pd.DataFrame):
            return data.head(1).values.tolist()[0]
        elif isinstance(data, list):
            return data[:1]
        return [0]

    def __preprocess_data(self, raw_path, quality_path):
        pass

    def __save_pmu_data(self, df_stat, df_top, process_name, save_path):
        os.makedirs(save_path, exist_ok=True)
        if df_stat is not None:
            df_stat.to_csv(os.path.join(save_path, f'perf_stat_{process_name}.csv'), index=False)
        if df_top is not None:
            df_top.to_csv(os.path.join(save_path, f'perf_top_{process_name}.csv'), index=False)

    # ----------------------------------------------------
    # 🌟 복구된 기계 가동 함수들 (operate_xxxx)
    # ----------------------------------------------------
    def operate_feeder(self, materials=None):
        return self.__feeder.operate_feeder(materials)

    def operate_drying_furnace(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()
        result = self.__drying_furnace.operate_drying_furnace(materials)
        df_stat, df_top = pmu.stop_and_collect()
        self.__save_pmu_data(df_stat, df_top, 'drying_furnace', 'Factory/PMU/Data')
        safe_val = self._extract_safe_value(result)
        self.send_message(address=120, message=str(len(safe_val)) + " - OK")
        return result

    def build_drying_furnace(self):
        self.__drying_furnace.build_drying_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_hardening_furnace(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()
        result = self.__hardening_furnace.operate_hardening_furnace(materials)
        df_stat, df_top = pmu.stop_and_collect()
        self.__save_pmu_data(df_stat, df_top, 'hardening_furnace', 'Factory/PMU/Data')
        safe_val = self._extract_safe_value(result)
        self.send_message(address=130, message=str(len(safe_val)) + " - OK")
        return result

    def build_hardening_furnace(self):
        self.__hardening_furnace.build_hardening_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_salt_bath(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()
        result = self.__salt_bath.operate_salt_bath(materials)
        df_stat, df_top = pmu.stop_and_collect()
        self.__save_pmu_data(df_stat, df_top, 'salt_bath', 'Factory/PMU/Data')
        safe_val = self._extract_safe_value(result)
        self.send_message(address=150, message=str(len(safe_val)) + " - OK")
        return result

    def build_salt_bath(self):
        self.__salt_bath.build_salt_bath(self.__train_dataset_path, self.__test_dataset_path)

    def operate_quenching(self, materials=None):
        result = self.__quenching.operate_quenching(materials)
        safe_val = self._extract_safe_value(result)
        self.send_message(address=140, message=str(len(safe_val)) + " - OK")
        return result

    def operate_washing(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()
        result = self.__washing.operate_washing(materials)
        df_stat, df_top = pmu.stop_and_collect()
        self.__save_pmu_data(df_stat, df_top, 'washing', 'Factory/PMU/Data')
        safe_val = self._extract_safe_value(result)
        self.send_message(address=110, message=str(len(safe_val)) + " - OK")
        return result

    def build_washing(self):
        self.__washing.build_washing(self.__train_dataset_path, self.__test_dataset_path)

    def operate_conveyor_belt(self, materials=None, port=8080):
        result = self.__conveyor_belt.operate_conveyor_belt(materials, target_port=port)
        safe_val = self._extract_safe_value(result)
        self.send_message(address=180, message=str(len(safe_val)) + " - OK")
        return result

    def operate_tempering_furnace(self, materials=None):
        result = self.__tempering_furnace.operate_tempering_furnace(materials)
        safe_val = self._extract_safe_value(result)
        self.send_message(address=160, message=str(len(safe_val)) + " - OK")
        return result

    def operate_rust_prevention(self, materials=None):
        result = self.__rust_prevention.operate_rust_prevention(materials)
        safe_val = self._extract_safe_value(result)
        self.send_message(address=170, message=str(len(safe_val)) + " - OK")
        return result