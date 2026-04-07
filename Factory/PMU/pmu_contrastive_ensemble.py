import os
import pandas as pd


class PmuContrastiveEnsemble:
    def __init__(self, heat_treatment_train_dataset_path, heat_treatment_test_dataset_path, pmu_data_dir_path):
        self.__heat_treatment_train_dataset = heat_treatment_train_dataset_path
        self.__heat_treatment_test_dataset = heat_treatment_test_dataset_path
        self.__pmu_data_dir = pmu_data_dir_path

    def __load_csv_to_df(self, file_path: str): # noqa
        """
        지정된 경로의 CSV 파일을 읽어 DataFrame으로 반환합니다.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        try:
            df = pd.read_csv(file_path, encoding='cp949') # noqa
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8') # noqa

        print(f"Successfully loaded: {os.path.basename(file_path)} (Rows: {len(df)})") # noqa
        return df

    def __load_all_csv_to_dict(self, dir_path: str): # noqa
        """
        지정된 디렉토리 내의 모든 CSV 파일을 읽어 딕셔너리로 반환합니다.
        결과 형식: {'파일명': DataFrame, ...}
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {dir_path}")

        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"입력된 경로가 디렉토리가 아닙니다: {dir_path}")

        csv_dict = {}
        # 디렉토리 내의 모든 파일 목록 가져오기
        file_list = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        if not file_list:
            print(f"[*] 주의: '{dir_path}' 폴더에 CSV 파일이 없습니다.")
            return csv_dict

        for file_name in file_list:
            file_path = os.path.join(dir_path, file_name)
            # 확장자를 제외한 파일명을 키(key)로 사용
            dict_key = os.path.splitext(file_name)[0]

            try:
                # 한글 깨짐 방지를 위해 cp949 우선 시도 후 utf-8 시도
                try:
                    df = pd.read_csv(file_path, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='utf-8')

                csv_dict[dict_key] = df
                print(f"Successfully loaded into dict: {file_name} (Rows: {len(df)})")

            except Exception as e:
                print(f"[!] '{file_name}' 로드 중 에러 발생: {e}")

        return csv_dict

    def prepare_datasets(self):
        pinn_train_df = self.__load_csv_to_df(self.__heat_treatment_train_dataset)
        pinn_test_df = self.__load_csv_to_df(self.__heat_treatment_test_dataset)
        pmu_data_dict = self.__load_all_csv_to_dict(self.__pmu_data_dir)

        print('김윤집')
        print(pmu_data_dict)
        print('김윤집')