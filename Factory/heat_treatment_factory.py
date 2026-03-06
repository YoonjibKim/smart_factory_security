from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Matrials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False):
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag)

    def _operate_heat_treatment(self): # noqa
        print("열처리 공정 시작")

        raw_part = self.__materials.load_pure_metal()
        self.__machine.operate_conveyor_belt(raw_part)
        self.__machine.operate_drying_furnace() # PINN
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_hardening_furnace() # PINN
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_salt_bath() # PINN
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_quenching()
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_washing() # PINN
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_tempering_furnace()
        self.__machine.operate_conveyor_belt()
        self.__machine.operate_rust_prevention()
        self.__machine.operate_conveyor_belt()
        processed_part = self.__materials.unload_hardened_part()

    def _build_heat_treatment_factory(self): # noqa
        print("열처리 공장 구축")
        # self.__machine.build_drying_furnace()
        # self.__machine.build_hardening_furnace()
        # self.__machine.build_salt_bath()
        # self.__machine.build_washing()
