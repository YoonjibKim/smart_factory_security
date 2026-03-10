from Factory.factory import Factory

if __name__ == '__main__':
    heat_treatment_factory = Factory(preprocess_dataset_flag=False, train_pinn_flag=False)
    # heat_treatment_factory.build_heat_treatment_factory()
    # heat_treatment_factory.operate_heat_treatment_factory()
    heat_treatment_factory.build_heat_treatment_defect_predictor()
    heat_treatment_factory.operate_heat_treatment_defect_predictor()