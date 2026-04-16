from Factory.factory import Factory


if __name__ == '__main__':
    heat_treatment_factory = Factory(preprocess_dataset_flag=False, train_pinn_flag=False)
    # heat_treatment_factory.build_heat_treatment_factory(is_pce=False)
    # heat_treatment_factory.build_heat_treatment_factory(is_pce=True)

    try:
        heat_treatment_factory.operate_heat_treatment_factory()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 열처리 공정(서버)이 강제 종료되었습니다.")

    # heat_treatment_factory.build_heat_treatment_defect_predictor()
    # heat_treatment_factory.operate_heat_treatment_defect_predictor()