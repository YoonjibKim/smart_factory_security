import pandas as pd


class HeatTreatmentMaterials:
    def __init__(self):
        pass

    def load_pure_metal(self):  # noqa
        file_path = 'Factory/PINN/Dataset/Heat_Treatment/Processed/test_data.csv'

        try:
            # CSV 파일을 DataFrame으로 변환
            df = pd.read_csv(file_path, encoding='cp949')
            print(f"[*] 원소재(Test Data) 로드 완료: 총 {len(df)}행 데이터가 투입됩니다.")
            return df

        except FileNotFoundError:
            print(f"[!] 에러: CSV 파일을 찾을 수 없습니다. 경로를 확인해 주세요.\n -> {file_path}")
            return None
        except Exception as e:
            print(f"[!] 데이터 로드 중 알 수 없는 에러 발생: {e}")
            return None

    def unload_hardened_part(self, materials=None): # noqa
        return materials