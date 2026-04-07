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
import pandas as pd
import os


class HeatTreatmentMachine(PMU):
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        PMU.__init__(self)
        raw_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/품질전처리후데이터.csv"
        quality_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/열처리_품질데이터.xlsx"
        processed_dir_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/"
        pinn_model_path = "Factory/Machine/Heat_Treatment/PINN/"

        self.__drying_furnace = DryingFurnace(train_pinn_flag, pinn_model_path)
        self.__hardening_furnace = HardeningFurnace(train_pinn_flag, pinn_model_path)
        self.__quenching = Quenching()
        self.__rust_prevention = RustPrevention()
        self.__salt_bath = SaltBath(train_pinn_flag, pinn_model_path)
        self.__tempering_furnace = TemperingFurnace()
        self.__washing = Washing(train_pinn_flag, pinn_model_path)
        self.__conveyor_belt = ConveyorBelt()
        self.__feeder = Feeder()

        if preprocess_dataset_flag:
            self.__preprocess_heat_treatment_dataset(raw_dataset_path, quality_dataset_path, processed_dir_path)

        self.__train_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/train_data.csv"
        self.__test_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/test_data.csv"

        self.__pmu_save_dir_path = "Factory/PMU/Data"

    class MachineType(Enum):
        DRYING = "drying_furnace"
        HARDENING = "hardening_furnace"
        SALT_BATH = "salt_bath"
        WASHING = "washing"

    def get_pmu_save_dir_path(self):
        return self.__pmu_save_dir_path

    def get_target_columns(self, machine_type: MachineType):
        if machine_type == self.MachineType.DRYING:
            return self.__drying_furnace.get_target_columns()
        elif machine_type == self.MachineType.HARDENING:
            return self.__hardening_furnace.get_target_columns()
        elif machine_type == self.MachineType.SALT_BATH:
            return self.__salt_bath.get_target_columns()
        elif machine_type == self.MachineType.WASHING:
            return self.__washing.get_target_columns()
        else:
            return None

    def get_train_dataset_path(self):
        return self.__train_dataset_path

    def get_test_dataset_path(self):
        return self.__test_dataset_path

    def __save_pmu_data(self, df_stat, df_top, type, base_dir): # noqa
        """
        df_stat와 df_top 데이터프레임을 type에 따라 구분하여 지정된 경로에 CSV로 저장합니다.
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Directory created: {base_dir}")

        stat_path = os.path.join(base_dir, f"perf_stat_{type}.csv")
        top_path = os.path.join(base_dir, f"perf_top_{type}.csv")

        try:
            df_stat.to_csv(stat_path, index=False, encoding='utf-8-sig')
            df_top.to_csv(top_path, index=False, encoding='utf-8-sig')

            print(f"Successfully saved ({type}):")
            print(f" - {stat_path}")
            print(f" - {top_path}")

        except Exception as e:
            print(f"Error saving data ({type}): {e}")

    def __preprocess_heat_treatment_dataset(self, raw_dataset_path, quality_dataset_path, merged_dir_path):  # noqa
        print(f"원본 데이터셋 로드: {raw_dataset_path}")

        raw_dataset_df = pd.read_csv(raw_dataset_path, encoding="cp949", engine="python") # noqa

        print(f"품질 데이터셋 로드: {quality_dataset_path}")
        quality_dataset_df = pd.read_excel(quality_dataset_path)

        raw_dataset_df['배정번호'] = raw_dataset_df['배정번호'].astype(str) # noqa
        quality_dataset_df['배정번호'] = quality_dataset_df['배정번호'].astype(str)

        target_columns = ['배정번호', '작업일', '양품수량', '불량수량', '총수량']
        quality_subset = quality_dataset_df[target_columns]

        df_merged = pd.merge(raw_dataset_df, quality_subset, on='배정번호', how='left') # noqa
        df_merged['불량률'] = (df_merged['불량수량'] / df_merged['총수량']).fillna(0)

        os.makedirs(merged_dir_path, exist_ok=True)

        merged_dataset_path = os.path.join(merged_dir_path, 'merged_data.csv')
        df_merged.to_csv(merged_dataset_path, index=False, encoding="cp949")
        print(f"병합 및 불량률 추가 완료. 파일 저장됨: {merged_dataset_path}")

        unique_batches = df_merged['배정번호'].dropna().unique()

        split_idx = int(len(unique_batches) * 0.8)
        train_batches = unique_batches[:split_idx]
        test_batches = unique_batches[split_idx:]

        train_df = df_merged[df_merged['배정번호'].isin(train_batches)].reset_index(drop=True)
        test_df = df_merged[df_merged['배정번호'].isin(test_batches)].reset_index(drop=True)

        train_df.to_csv(os.path.join(merged_dir_path, 'train_data.csv'), index=False, encoding='cp949')
        test_df.to_csv(os.path.join(merged_dir_path, 'test_data.csv'), index=False, encoding='cp949')

        print(f"[데이터 분할 완료] Train 배차 수: {len(train_batches)}개, Test 배차 수: {len(test_batches)}개")
        print(f"Train 데이터 크기: {train_df.shape}, Test 데이터 크기: {test_df.shape}")
        print(f"파일 저장 완료: {os.path.abspath(merged_dir_path)} 폴더를 확인하세요.")

    def operate_drying_furnace(self, materials=None):
        # top_n=5 가 기본값으로 설정되어 있습니다. (PMU(top_n=10) 등으로 변경 가능)
        pmu = PMU()

        pmu._start_perf_record()
        pmu._start_perf_stat()

        result = self.__drying_furnace.operate_drying_furnace(materials)

        df_stat, df_top = pmu.stop_and_collect()

        print("\n[PERF STAT DATA]")
        print(df_stat)

        print("\n[PERF TOP DATA]")
        print(df_top)
        self.__save_pmu_data(df_stat, df_top, 'drying_furnace', self.__pmu_save_dir_path)

        return result

    def build_drying_furnace(self):
        self.__drying_furnace.build_drying_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_hardening_furnace(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()

        result = self.__hardening_furnace.operate_hardening_furnace(materials)

        df_stat, df_top = pmu.stop_and_collect()
        print("\n[PERF STAT DATA]")
        print(df_stat)
        print("\n[PERF TOP DATA]")
        print(df_top)
        self.__save_pmu_data(df_stat, df_top, 'hardening_furnace', 'Factory/PMU/Data')

        return result

    def build_hardening_furnace(self):
        self.__hardening_furnace.build_hardening_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_quenching(self, materials=None):
        result = self.__quenching.operate_quenching(materials)
        return result

    def build_quenching(self):
        self.__quenching.build_quenching()

    def operate_rust_prevention(self, materials=None):
        result = self.__rust_prevention.operate_rust_prevention(materials)
        return result

    def build_rust_prevention(self):
        self.__rust_prevention.build_rust_prevention()

    def operate_salt_bath(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()

        result = self.__salt_bath.operate_salt_bath(materials)

        df_stat, df_top = pmu.stop_and_collect()
        print("\n[PERF STAT DATA]")
        print(df_stat)
        print("\n[PERF TOP DATA]")
        print(df_top)
        self.__save_pmu_data(df_stat, df_top, 'salt_bath', 'Factory/PMU/Data')

        return result

    def build_salt_bath(self):
        self.__salt_bath.build_salt_bath(self.__train_dataset_path, self.__test_dataset_path)

    def operate_tempering_furnace(self, materials=None):
        result = self.__tempering_furnace.operate_tempering_furnace(materials)
        return result

    def build_tempering_furnace(self):
        self.__tempering_furnace.build_tempering_furnace()

    def operate_washing(self, materials=None):
        pmu = PMU()
        pmu._start_perf_record()
        pmu._start_perf_stat()

        result = self.__washing.operate_washing(materials)

        df_stat, df_top = pmu.stop_and_collect()
        print("\n[PERF STAT DATA]")
        print(df_stat)
        print("\n[PERF TOP DATA]")
        print(df_top)
        self.__save_pmu_data(df_stat, df_top, 'washing', 'Factory/PMU/Data')

        return result

    def build_washing(self):
        self.__washing.build_washing(self.__train_dataset_path, self.__test_dataset_path)

    def operate_conveyor_belt(self, materials=None):
        result = self.__conveyor_belt.operate_conveyor_belt(materials)
        return result

    def build_conveyor_belt(self):
        self.__conveyor_belt.build_conveyor_belt()

    def operate_feeder(self, materials=None):
        result = self.__feeder.operate_feeder(materials)
        return result

    def build_feeder(self):
        self.__feeder.build_feeder()