from Factory.Machine.Heat_Treatment.conveyor_belt import ConveyorBelt
from Factory.Machine.Heat_Treatment.drying_furnace import DryingFurnace
from Factory.Machine.Heat_Treatment.feeder import Feeder
from Factory.Machine.Heat_Treatment.hardening_furnace import HardeningFurnace
from Factory.Machine.Heat_Treatment.quenching import Quenching
from Factory.Machine.Heat_Treatment.rust_prevention import RustPrevention
from Factory.Machine.Heat_Treatment.salt_bath import SaltBath
from Factory.Machine.Heat_Treatment.tempering_furnace import TemperingFurnace
from Factory.Machine.Heat_Treatment.washing import Washing
import pandas as pd
import os


class HeatTreatmentMachine(DryingFurnace, HardeningFurnace, Quenching, RustPrevention, SaltBath, TemperingFurnace,
                           Washing, ConveyorBelt, Feeder):
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        raw_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/품질전처리후데이터.csv"
        quality_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Raw/열처리_품질데이터.xlsx"
        processed_dir_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/"
        pinn_model_path = "Factory/Machine/Heat_Treatment/PINN/"

        DryingFurnace.__init__(self, train_pinn_flag, pinn_model_path)
        HardeningFurnace.__init__(self, train_pinn_flag, pinn_model_path)
        Quenching.__init__(self)
        RustPrevention.__init__(self)
        SaltBath.__init__(self, train_pinn_flag, pinn_model_path)
        TemperingFurnace.__init__(self)
        Washing.__init__(self, train_pinn_flag, pinn_model_path)
        ConveyorBelt.__init__(self)
        Feeder.__init__(self)

        if preprocess_dataset_flag:
            self.__preprocess_heat_treatment_dataset(raw_dataset_path, quality_dataset_path, processed_dir_path)

        self.__train_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/train_data.csv"
        self.__test_dataset_path = "Factory/PINN/Dataset/Heat_Treatment/Processed/test_data.csv"

    def __preprocess_heat_treatment_dataset(self, raw_dataset_path, quality_dataset_path, merged_dir_path):  # noqa
        print(f"원본 데이터셋 로드: {raw_dataset_path}")

        raw_dataset_df = pd.read_csv(raw_dataset_path, encoding="cp949", engine="python")

        print(f"품질 데이터셋 로드: {quality_dataset_path}")
        quality_dataset_df = pd.read_excel(quality_dataset_path)

        raw_dataset_df['배정번호'] = raw_dataset_df['배정번호'].astype(str)
        quality_dataset_df['배정번호'] = quality_dataset_df['배정번호'].astype(str)

        target_columns = ['배정번호', '작업일', '양품수량', '불량수량', '총수량']
        quality_subset = quality_dataset_df[target_columns]

        df_merged = pd.merge(raw_dataset_df, quality_subset, on='배정번호', how='left')
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

    def operate_drying_furnace(self):
        self._operate_drying_furnace()

    def build_drying_furnace(self):
        self._build_drying_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_hardening_furnace(self):
        self._operate_hardening_furnace()

    def build_hardening_furnace(self):
        self._build_hardening_furnace(self.__train_dataset_path, self.__test_dataset_path)

    def operate_quenching(self):
        self._operate_quenching()

    def build_quenching(self):
        self._build_quenching()

    def operate_rust_prevention(self):
        self._operate_rust_prevention()

    def build_rust_prevention(self):
        self._build_rust_prevention()

    def operate_salt_bath(self):
        self._operate_salt_bath()

    def build_salt_bath(self):
        self._build_salt_bath(self.__train_dataset_path, self.__test_dataset_path)

    def operate_tempering_furnace(self):
        self._operate_tempering_furnace()

    def build_tempering_furnace(self):
        self._build_tempering_furnace()

    def operate_washing(self):
        self._operate_washing()

    def build_washing(self):
        self._build_washing(self.__train_dataset_path, self.__test_dataset_path)

    def operate_conveyor_belt(self, parts=None):
        self._operate_conveyor_belt()

    def build_conveyor_belt(self):
        self._build_conveyor_belt()

    def operate_feeder(self):
        self._operate_feeder()

    def build_feeder(self):
        self._build_feeder()