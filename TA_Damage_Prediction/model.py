import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import pandas as pd

# CSV 파일 경로
file_path = r"C:\Users\SSU\Desktop\Data_Ana\acc.csv"

# CSV 파일 읽기
try:
    data = pd.read_csv(file_path)
    print("데이터 로드 성공!")
except Exception as e:
    print(f"파일을 읽는 중 오류 발생: {e}")

# 기존 열 이름 리스트
original_columns = [
    'acc_year', 'occrrnc_dt', 'dght_cd', 'occrrnc_day_cd', 
    'dth_dnv_cnt', 'injpsn_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 
    'wnd_dnv_cnt', 'occrrnc_lc_sido_cd', 'occrrnc_lc_sgg_cd', 
    'acc_ty_lclas_cd', 'acc_ty_mlsfc_cd', 'acc_ty_cd', 'aslt_vtr_cd', 
    'road_frm_lclas_cd', 'road_frm_cd', 'wrngdo_isrty_vhcty_lclas_cd', 
    'dmge_isrty_vhcty_lclas_cd', 'occrrnc_ls_x_crd', 'occrrnc_ls_y_crd', 
    'lo_crd', 'la_crd'
]

# 변경할 열 이름 리스트
new_columns = [
    '사고년도', '월일시', '주야구분코드', '요일코드', '사망자수', '부상자수', 
    '중상자수', '경상자수', '부상신고자수', '위치 시도코드', '위치 시군구코드', 
    '사고유형 대분류코드', '사고유형 중분류코드', '사고유형 코드', 
    '가해자 법규위반코드', '도로형태 대분류코드', '도로형태 코드', 
    '가해당사자 차종별대분류코드', '피해당사자 차종별대분류코드', 
    '위치 X좌표', '위치 Y좌표', '경도좌표', '위도좌표'
]

# 열 이름 변경
data.columns = new_columns

# 결과 확인
print("열 이름 변경 완료:")
print(data.columns)

# 삭제할 열 이름 리스트
columns_to_drop = ['월일시','위치 X좌표', '위치 Y좌표', '경도좌표', '위도좌표']

# 열 삭제
data = data.drop(columns=columns_to_drop)

# 결과 확인
print("열 삭제 후 데이터프레임:")
print(data.head())
# 사고 피해 정도 열 생성 
# 사고 피해 정도 계산
data['사고 피해 정도'] = (data['사망자수'] * 10) + (data['중상자수'] * 5) + (data['경상자수'] * 1)

# 결과 확인
print(data[['사망자수', '중상자수', '경상자수', '사고 피해 정도']].head())


################# 여기까지 전처리 

# '사고 피해 정도'를 범주형으로 변환하는 함수
def categorize_damage(severity):
    if severity <= 10:
        return '낮음'
    elif severity <= 20:
        return '중간'
    else:
        return '높음'

# '사고 피해 정도'를 범주화
data['피해_범주'] = data['사고 피해 정도'].apply(categorize_damage)

# 입력 변수와 출력 변수 설정
X = data[['주야구분코드', '요일코드', '위치 시도코드', '위치 시군구코드', 
          '사고유형 대분류코드', '사고유형 중분류코드', '사고유형 코드',
          '가해자 법규위반코드', '도로형태 대분류코드', '도로형태 코드', 
          '가해당사자 차종별대분류코드', '피해당사자 차종별대분류코드']]
y = data['피해_범주']

# 범주형 변수를 One-Hot Encoding으로 변환
X = pd.get_dummies(X, columns=X.columns)

# 목표 변수(Label) 숫자형 변환
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # '낮음', '중간', '높음' -> 0, 1, 2

# 입력 데이터 타입 변환 (SMOTE 및 모델 호환성을 위해 float 타입으로 변환)
X = X.astype(float)

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# SMOTE 적용 전 레이블 분포 출력
print("SMOTE 적용 전 레이블 분포:", Counter(y_train))

# SMOTE를 이용한 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# SMOTE 적용 후 레이블 분포 출력
print("SMOTE 적용 후 레이블 분포:", Counter(y_train_resampled))


# 데이터 정규화 (StandardScaler)
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# 결과 저장용 함수
def evaluate_model(model, model_name):
    model.fit(X_train_resampled_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{model_name} 정확도:", accuracy_score(y_test, y_pred))
    print(f"\n{model_name} 분류 리포트:\n", classification_report(y_test, y_pred))

# 1. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
evaluate_model(rf_model, "Random Forest")

# 2. Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
evaluate_model(gb_model, "Gradient Boosting")

# 3. LightGBM
lgbm_model = LGBMClassifier(random_state=42)
evaluate_model(lgbm_model, "LightGBM")

# 4. CatBoost
cat_model = CatBoostClassifier(verbose=0, random_state=42)
evaluate_model(cat_model, "CatBoost")

# 5. Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
evaluate_model(logistic_model, "Logistic Regression")