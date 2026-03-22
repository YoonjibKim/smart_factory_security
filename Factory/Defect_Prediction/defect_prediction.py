import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class DefectPrediction:
    def __init__(self, model_dir_path):
        self.__model_dir_path = model_dir_path

        # 🚀 [수정됨] 엣지 디바이스(NVIDIA GPU 없음) 환경을 고려한 자동 분기 처리
        # 환경 변수나 라이브러리 상태를 확인하기보다, try-except로 안전하게 CPU Fallback을 유도할 수 있도록
        # 기본값은 CPU 호환 모드로 초기화합니다. 학습(build) 시에만 GPU 설정을 주입하도록 변경합니다.
        self.model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            random_state=42,
            verbose=100
        )

        # 압축하기 전의 원본 센서 컬럼들
        self.base_features = [
            '소입1존 OP', '소입2존 OP', '소입3존 OP', '소입4존 OP',
            '소입로 온도 1 Zone', '소입로 온도 2 Zone',
            '솔트조 온도 1 Zone', '세정기'
        ]

        # 배차 단위로 압축된 후 AI가 학습할 최종 특성(Feature) 목록 (평균, 편차, 최대, 최소)
        self.features = [f"{col}_{stat}" for col in self.base_features for stat in ['mean', 'std', 'max', 'min']]

        # 우리가 최종적으로 맞혀야 하는 정답
        self.target = '불량률'

    def load_dataset(self, dataset_path): # noqa
        """
        한글 컬럼이 포함된 CSV 파일을 읽어서 DataFrame으로 반환합니다.
        """
        if not os.path.exists(dataset_path):
            print(f"[!] 에러: 파일을 찾을 수 없습니다 -> {dataset_path}")
            return None

        try:
            # 1. 일반적인 한글 Windows 환경 표준 인코딩 (가장 권장)
            df = pd.read_csv(dataset_path, encoding='cp949')
            return df
        except UnicodeDecodeError:
            try:
                # 2. cp949로 실패할 경우 UTF-8 시도
                df = pd.read_csv(dataset_path, encoding='utf-8')
                return df
            except UnicodeDecodeError:
                # 3. 드물게 발생하는 euc-kr 시도
                df = pd.read_csv(dataset_path, encoding='euc-kr')
                return df
        except Exception as e:
            print(f"[!] 데이터 로드 중 예상치 못한 에러 발생: {e}")
            return None

    def _aggregate_by_batch(self, df):
        """
        초 단위로 기록된 데이터를 '배정번호(배차)' 단위로 압축합니다.
        평균(mean), 편차(std), 최대(max), 최소(min) 값을 추출하여 강력한 AI 특성으로 변환합니다.
        """
        if '배정번호' not in df.columns:
            print("[!] 경고: '배정번호' 컬럼이 없어 배차 단위 압축을 수행할 수 없습니다. 원본을 유지합니다.")
            return df

        print(f"[*] 데이터를 '배차(배정번호)' 단위로 요약 및 압축합니다... (원본: {len(df)}행)")

        # 요약할 대상 (존재하는 센서 데이터만)
        sensor_cols = [col for col in self.base_features if col in df.columns]

        agg_funcs = {}
        for col in sensor_cols:
            agg_funcs[col] = ['mean', 'std', 'max', 'min']

        # 불량률은 1배차당 1개의 고정값이므로 평균을 내면 원래 값 그대로 유지됨
        if self.target in df.columns:
            agg_funcs[self.target] = ['mean']

        # 배정번호별로 그룹화 수행
        grouped = df.groupby('배정번호').agg(agg_funcs).reset_index()

        # 컬럼 이름 평탄화 (예: '소입1존 OP' + 'mean' -> '소입1존 OP_mean')
        new_cols = []
        for col in grouped.columns:
            if col[1] == '':
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        grouped.columns = new_cols

        # 결측치(데이터가 1줄이라 편차(std)가 NaN이 되는 경우 등)를 0으로 안전하게 채움
        grouped = grouped.fillna(0)

        # 타겟 이름 복구 ('불량률_mean' -> '불량률')
        if f"{self.target}_mean" in grouped.columns:
            grouped = grouped.rename(columns={f"{self.target}_mean": self.target})

        print(f"[*] 압축 완료! 총 {len(grouped)}개의 배차 데이터로 변환되었습니다.")
        return grouped

    def build_defect_predictor(self, train_df, test_df):
        """
        [학습 단계]
        공장의 과거 데이터(train_df)를 보고 캣부스트 모델을 GPU로 초고속 학습시킨 뒤 저장합니다.
        """
        print("\n================ [불량률 예측 AI (CatBoost GPU) 학습 시작] ================")

        if train_df is None or train_df.empty:
            print("[!] 에러: 학습 데이터(train_df)가 없습니다.")
            return

        # 데이터를 배차 단위로 압축!
        train_agg = self._aggregate_by_batch(train_df)

        # 1. 실제 데이터에 존재하는 컬럼만 안전하게 필터링
        available_features = [col for col in self.features if col in train_agg.columns]
        if not available_features:
            print("[!] 에러: 학습에 필요한 압축 센서 컬럼을 찾을 수 없습니다.")
            return

        # 2. 결측치(NaN)가 있는 불량 데이터 제거
        train_clean = train_agg.dropna(subset=available_features + [self.target]).copy()

        X = train_clean[available_features]
        y = train_clean[self.target]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. 모델 학습 수행 (🚨 GPU 강제 할당을 학습 함수 내부에서 처리)
        print(f"[*] 학습 데이터 세팅 완료 (Train: {len(X_train)}배차, Val: {len(X_val)}배차).")
        print(f"[*] RTX 5090 가속 및 🚨조기 종료(Early Stopping)🚨 모드로 학습을 진행합니다...")

        # 기존 self.model을 GPU 옵션이 추가된 모델로 새로 덮어씌움 (학습용 서버에서만 실행됨)
        self.model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            random_state=42,
            verbose=100,
            task_type='GPU',
            devices='0'
        )

        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )

        # 4. 학습된 모델 저장
        os.makedirs(self.__model_dir_path, exist_ok=True)
        model_save_path = os.path.join(self.__model_dir_path, "catboost_defect_model.cbm")
        self.model.save_model(model_save_path)
        print(f"[*] 학습 완료! 모델이 성공적으로 저장되었습니다 -> {model_save_path}")

        # 5. 테스트 데이터 채점
        if test_df is not None and not test_df.empty:
            test_agg = self._aggregate_by_batch(test_df)
            test_clean = test_agg.dropna(subset=available_features + [self.target]).copy()
            if not test_clean.empty:
                X_test = test_clean[available_features]
                y_test = test_clean[self.target]

                y_pred = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                print(f"[*] 🎯 방금 학습한 모델의 Test 데이터(배차 단위) MAE (평균 오차율): {mae:.5f}")

    def operate_defect_predictor(self, test_df):
        """
        [추론/가동 단계]
        저장된 GPU 모델을 불러와서 현재 공정에 투입된 자재(test_df)의 불량률을 예측합니다.
        (엣지 디바이스에서는 자동으로 CPU를 사용하여 추론합니다.)
        """
        print("\n================ [불량률 예측 AI (CatBoost Edge CPU) 추론 가동] ================")

        model_load_path = os.path.join(self.__model_dir_path, "catboost_defect_model.cbm")

        # 1. 모델 불러오기
        # (cbm 확장자 파일은 GPU에서 학습되었더라도 CPU 모드로 로드 및 추론이 가능합니다)
        if os.path.exists(model_load_path):
            self.model.load_model(model_load_path)
            print(f"[*] 엣지 환경에 최적화된 캣부스트 모델(CPU) 로드 완료 -> {model_load_path}")
        else:
            print(f"[!] 에러: 모델을 찾을 수 없습니다. 먼저 GPU 서버에서 build를 실행해 주세요 -> {model_load_path}")
            return test_df

        if test_df is None or test_df.empty:
            print("[!] 경고: 예측할 데이터(test_df)가 없어 추론을 건너뜁니다.")
            return test_df

        # 테스트 데이터 압축
        test_agg = self._aggregate_by_batch(test_df)

        # 2. 예측 수행을 위한 특성 필터링
        available_features = [col for col in self.features if col in test_agg.columns]
        X_test = test_agg[available_features].fillna(0)

        print(f"[*] 입력된 공정 데이터({len(X_test)}개 배차)의 최종 불량률 초고속 예측을 시작합니다...")
        y_pred = self.model.predict(X_test)

        # 3. 결과 정리 및 출력
        result_df = test_agg.copy()
        result_df['AI_예측_불량률'] = y_pred

        print("\n[*] 🎯 예측 완료! (배차별 결과 종합)")

        cols_to_show = []
        if '배정번호' in result_df.columns:
            cols_to_show.append('배정번호')

        if '불량률' in result_df.columns:
            cols_to_show.append('불량률')
            cols_to_show.append('AI_예측_불량률')

            result_df['오차(절댓값)'] = (result_df['불량률'] - result_df['AI_예측_불량률']).abs()
            cols_to_show.append('오차(절댓값)')

            threshold = 0.001
            result_df['AI_품질판정'] = np.where(result_df['AI_예측_불량률'] >= threshold, '🔴불량 경고', '🟢정상 양품')
            cols_to_show.append('AI_품질판정')
        else:
            cols_to_show.append('AI_예측_불량률')

        pd.set_option('display.max_rows', 100)
        print(result_df[cols_to_show].head(100))
        print("=======================================================================\n")

        return result_df