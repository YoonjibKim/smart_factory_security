from Factory.heat_treatment_factory import HeatTreatmentFactory


class Factory(HeatTreatmentFactory):
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        HeatTreatmentFactory.__init__(self, preprocess_dataset_flag, train_pinn_flag)

    def build_heat_treatment_factory(self):
        self._build_heat_treatment_factory()

    def operate_heat_treatment_factory(self):
        self._operate_heat_treatment()
