from Factory.Defect_Prediction.defect_prediction import DefectPrediction
from Factory.heat_treatment_factory import HeatTreatmentFactory


class Factory(HeatTreatmentFactory):
    def __init__(self, preprocess_dataset_flag=False, train_pinn_flag=False):
        HeatTreatmentFactory.__init__(self, preprocess_dataset_flag, train_pinn_flag)
        self.__heat_treatment_defect_predictor = DefectPrediction(model_dir_path="Factory/Defect_Prediction/Model")

    def build_heat_treatment_factory(self):
        self._build_heat_treatment_factory()

    def operate_heat_treatment_factory(self):
        self._operate_heat_treatment()

    def build_heat_treatment_defect_predictor(self):
        train_df = self.__heat_treatment_defect_predictor.load_dataset('Factory/PINN/Dataset/Heat_Treatment/'
                                                                       'Processed/train_data.csv')
        test_df = self.__heat_treatment_defect_predictor.load_dataset('Factory/PINN/Dataset/Heat_Treatment/'
                                                                      'Processed/test_data.csv')
        self.__heat_treatment_defect_predictor.build_defect_predictor(train_df, test_df)

    def operate_heat_treatment_defect_predictor(self):
        test_df = self.__heat_treatment_defect_predictor.load_dataset('Factory/PINN/Dataset/Heat_Treatment/'
                                                                      'Processed/test_data.csv')
        self.__heat_treatment_defect_predictor.operate_defect_predictor(test_df)

