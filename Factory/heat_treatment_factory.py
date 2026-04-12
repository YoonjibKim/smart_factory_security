from Factory.Machine.Heat_Treatment.heat_treatment_machine import HeatTreatmentMachine
from Factory.Materials.Heat_Treatment.heat_treatment_materials import HeatTreatmentMaterials
from Factory.PMU.pmu_contrastive_ensemble import PmuContrastiveEnsemble


class HeatTreatmentFactory:
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        # 자재 및 머신 객체 초기화 (메인 메모리에 안전하게 적재됨)
        self.__materials = HeatTreatmentMaterials()
        self.__machine = HeatTreatmentMachine(preprocess_dataset_flag, train_pinn_flag)

    def _build_heat_treatment_factory(self, is_pce=False):
        if not is_pce:
            print("\n================ 열처리 공장 구축 시작 ================")
            print("[*] 각 공정의 PINN 모델 및 설비를 순차적으로 초기화합니다...")

            # 각 공정의 모델을 순차적으로 빌드(로드)합니다.
            self.__machine.build_drying_furnace()
            self.__machine.build_hardening_furnace()
            self.__machine.build_salt_bath()
            self.__machine.build_washing()

            print("[*] 열처리 공장 구축 및 메모리 적재 완료.\n")
        else:
            # self._operate_heat_treatment()
            pce = PmuContrastiveEnsemble(heat_treatment_train_dataset_path=self.__machine.get_train_dataset_path(),
                                         heat_treatment_test_dataset_path=self.__machine.get_test_dataset_path(),
                                         pmu_data_dir_path=self.__machine.get_pmu_save_dir_path())

            target_columns_dict = {
                self.__machine.MachineType.DRYING:
                    self.__machine.get_target_columns(machine_type=self.__machine.MachineType.DRYING),
                self.__machine.MachineType.HARDENING:
                    self.__machine.get_target_columns(machine_type=self.__machine.MachineType.HARDENING),
                self.__machine.MachineType.WASHING:
                    self.__machine.get_target_columns(machine_type=self.__machine.MachineType.WASHING),
                self.__machine.MachineType.SALT_BATH:
                    self.__machine.get_target_columns(machine_type=self.__machine.MachineType.SALT_BATH)
            }

            # pce.prepare_datasets(target_columns_dict)
            # pce.generate_model(target_columns_dict)
            pce.generate_edge_model(target_columns_dict)

    def _operate_heat_treatment(self):
        print("\n================ 열처리 공정 시작 ================")

        # 1. 금속 재료 투입
        raw_materials = self.__materials.load_pure_metal()
        raw_materials = self.__machine.operate_feeder(raw_materials)
        raw_materials = self.__machine.operate_conveyor_belt(raw_materials)

        # 2. 건조로 공정
        drying_furnace_materials = self.__machine.operate_drying_furnace(raw_materials)
        drying_furnace_materials = self.__machine.operate_conveyor_belt(drying_furnace_materials)

        # 3. 소입로 공정
        hardening_furnace_materials = self.__machine.operate_hardening_furnace(drying_furnace_materials)
        hardening_furnace_materials = self.__machine.operate_conveyor_belt(hardening_furnace_materials)

        # 4. 솔트조 공정
        # salt_bath_furnace_materials = self.__machine.operate_salt_bath(hardening_furnace_materials)
        # salt_bath_furnace_materials = self.__machine.operate_conveyor_belt(salt_bath_furnace_materials)

        # 5. 퀜칭 공정
        # quenching_furnace_materials = self.__machine.operate_quenching(salt_bath_furnace_materials)
        # quenching_furnace_materials = self.__machine.operate_conveyor_belt(quenching_furnace_materials)
        #
        # 6. 세정기 공정
        # washing_furnace_materials = self.__machine.operate_washing(quenching_furnace_materials)
        # washing_furnace_materials = self.__machine.operate_conveyor_belt(washing_furnace_materials)
        #
        # 7. 소려로 공정
        # tempering_furnace_furnace_materials = self.__machine.operate_tempering_furnace(washing_furnace_materials)
        # tempering_furnace_furnace_materials = self.__machine.operate_conveyor_belt(tempering_furnace_furnace_materials)
        #
        # 8. 방청 공정
        # rust_prevention_furnace_materials = self.__machine.operate_rust_prevention(tempering_furnace_furnace_materials)
        # rust_prevention_furnace_materials = self.__machine.operate_conveyor_belt(rust_prevention_furnace_materials)
        #
        # 9. 부품 가공 완료 및 배출
        # processed_part = self.__materials.unload_hardened_part(rust_prevention_furnace_materials)
        # print(processed_part)
        print("================ 열처리 공정 종료 ================\n")