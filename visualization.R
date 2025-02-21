
# 강원도 교통사고 데이터 ------------------------------------------------------------

# library -----------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(randomForest)
library(ggplot2)
library(ggmap)
library(leaflet)
library(RColorBrewer)
library(pROC)
library(caret)

# data loading ------------------------------------------------------------

TA <- read.csv("병합된_교통사고통계.csv")

# data cleaning -----------------------------------------------------------

TA2 <- TA %>%
  rename(death_cnt = dth_dnv_cnt,
         fse_cnt = se_dnv_cnt,
         msl_cnt = sl_dnv_cnt,
         occDate = occrrnc_dt,
         accType = acc_ty_lclas_cd,
         accTypeD = acc_ty_cd,
         carFLg = aslt_vtr_cd,
         carClassF = wrngdo_isrty_vhcty_lclas_cd,
         carClassVic = dmge_isrty_vhcty_lclas_cd,
         road_form_class = road_frm_lclas_cd,
         road_formD = road_frm_cd,
         sigungu = occrrnc_lc_sgg_cd)


TA2 <- TA2 %>%
  mutate(
    occDate = as.character(occDate),
    year = substr(occDate, 1, 4), 
    month = substr(occDate, 5, 6),
    day = substr(occDate, 7, 8),  
    hour = substr(occDate, 9, 10), 
    date = as.Date(paste(year, month, day, sep = "-"))
  )

# 연도별 휴일 기간 지정
holiday_periods <- data.frame(
  start_date = as.Date(c(
    "2012-01-01", "2012-01-21", "2012-09-29", "2012-12-24", # 2012년 휴일
    "2013-01-01", "2013-02-09", "2013-09-18", "2013-12-24", # 2013년 휴일
    "2014-01-01", "2014-01-30", "2014-09-06", "2014-12-24", # 2014년 휴일
    "2015-01-01", "2015-02-18", "2015-09-26", "2015-12-24", # 2015년 휴일
    "2016-01-01", "2016-02-06", "2016-09-14", "2016-12-24", # 2016년 휴일
    "2017-01-01", "2017-01-27", "2017-09-30", "2017-12-24", # 2017년 휴일
    "2018-01-01", "2018-02-15", "2018-09-22", "2018-12-24", # 2018년 휴일
    "2019-01-01", "2019-02-02", "2019-09-12", "2019-12-24", # 2019년 휴일
    "2020-01-01", "2020-01-24", "2020-09-30", "2020-12-24", # 2020년 휴일
    "2021-01-01", "2021-02-11", "2021-09-18", "2021-12-24", # 2021년 휴일
    "2022-01-01", "2022-01-30", "2022-09-09", "2022-12-24", # 2022년 휴일
    "2023-01-01", "2023-01-21", "2023-09-28", "2023-12-24"  # 2023년 휴일
  )),
  end_date = as.Date(c(
    "2012-01-02", "2012-01-24", "2012-10-01", "2012-12-31", # 2012년 종료일
    "2013-01-02", "2013-02-11", "2013-09-22", "2013-12-31", # 2013년 종료일
    "2014-01-02", "2014-02-02", "2014-09-10", "2014-12-31", # 2014년 종료일
    "2015-01-02", "2015-02-22", "2015-09-29", "2015-12-31", # 2015년 종료일
    "2016-01-02", "2016-02-10", "2016-09-18", "2016-12-31", # 2016년 종료일
    "2017-01-02", "2017-01-30", "2017-10-09", "2017-12-31", # 2017년 종료일
    "2018-01-02", "2018-02-18", "2018-09-26", "2018-12-31", # 2018년 종료일
    "2019-01-02", "2019-02-06", "2019-09-15", "2019-12-31", # 2019년 종료일
    "2020-01-02", "2020-01-27", "2020-10-04", "2020-12-31", # 2020년 종료일
    "2021-01-02", "2021-02-14", "2021-09-22", "2021-12-31", # 2021년 종료일
    "2022-01-02", "2022-02-02", "2022-09-12", "2022-12-31", # 2022년 종료일
    "2023-01-02", "2023-01-24", "2023-10-03", "2023-12-31"  # 2023년 종료일
  ))
)

# 휴일 여부 추가
TA2 <- TA2 %>%
  rowwise() %>%
  mutate(
    is_holiday = ifelse(
      any(date >= holiday_periods$start_date & date <= holiday_periods$end_date),
      1, 0
    )
  ) %>%
  ungroup()

# 결과 확인
head(TA2)


# factor형 독립변수로 변경 --------------------------------------------------------

TA2$accType <- factor(TA2$accType)
TA2$accTypeD <- factor(TA2$accTypeD)
TA2$carFLg <- factor(TA2$carFLg)
TA2$carClassF <- factor(TA2$carClassF)
TA2$carClassVic <- factor(TA2$carClassVic)
TA2$road_form_class <- factor(TA2$road_form_class)
TA2$road_formD <- factor(TA2$road_formD)

# factor형 독립변수 level 값 변경 -------------------------------------------------


#accType
summary(TA2$accType)

TA2 <- TA2 %>%
  mutate(accType = as.character(accType)) %>%
  mutate(
    accType = case_when(
      accType %in% "1" ~ "0", 
      accType %in% "2" ~ "1", 
      accType %in% "3" ~ "2", 
      accType %in% "4" ~ "3",
      TRUE ~ NA
    )
  )

TA2$accType <- factor(TA2$accType)
summary(TA2$accType)
sum(is.na(TA2$accType))


#accTypeD
summary(TA2$accTypeD)

TA2 <- TA2 %>%
  mutate(accTypeD = as.character(accTypeD)) %>%
  mutate(
    accTypeD = case_when(
      accTypeD %in% "01" ~ "0",
      accTypeD %in% "02" ~ "1",
      accTypeD %in% "03" ~ "2", 
      accTypeD %in% c("23", "Z1", "Z2") ~ "3",
      accTypeD %in% c("21", "22", "26", "32", "33") ~ "4", 
      accTypeD %in% "04" ~ "5",
      accTypeD %in% c("34", "35") ~ "6",
      accTypeD %in% c("31", "38", "39") ~ "7",
      accTypeD %in% c("05", "25", "37", "Z7", "Z8") ~ "8",
      accTypeD %in% c("36", "41", "Z4", "Z5", "Z6", "##") ~ "9",
      TRUE ~ NA
    )
  )

TA2$accTypeD <- factor(TA2$accTypeD)
summary(TA2$accTypeD)
sum(is.na(TA2$accTypeD))


#carFLg
summary(TA2$carFLg)

TA2 <- TA2 %>%
  mutate(carFLg = as.character(carFLg)) %>%
  mutate(
    carFLg = case_when(
      carFLg %in% "1" ~ "0", 
      carFLg %in% "2" ~ "1",
      carFLg %in% "3" ~ "2",
      carFLg %in% "4" ~ "3",
      carFLg %in% "5" ~ "4",
      carFLg %in% "6" ~ "5",
      carFLg %in% "7" ~ "6",
      carFLg %in% c("99", "##") ~ "7",
      TRUE ~ NA
    )
  )

TA2$carFLg <- factor(TA2$carFLg)
summary(TA2$carFLg)


#carClassF
summary(TA2$carClassF)

TA2 <- TA2 %>%
  mutate(carClassF = as.character(carClassF)) %>%
  mutate(
    carClassF = case_when(
      carClassF %in% "1" ~ "0", 
      carClassF %in% "2" ~ "1",
      carClassF %in% "3" ~ "2", 
      carClassF %in% c("4", "6") ~ "3", 
      carClassF %in% "5" ~ "4",
      carClassF %in% c("7", "8", "9") ~ "5",
      carClassF %in% c("10", "11") ~ "6",
      carClassF %in% "12" ~ "7",
      carClassF %in% c("98", "99", "Z1", "ZL", "##") ~ "8",
      TRUE ~ NA
    )
  )

TA2$carClassF <- factor(TA2$carClassF)
summary(TA2$carClassF)


#carClassVic
summary(TA2$carClassVic)

TA2 <- TA2 %>%
  mutate(carClassVic = as.character(carClassVic)) %>%
  mutate(
    carClassVic = case_when(
      carClassVic %in% "01" ~ "0", 
      carClassVic %in% "02" ~ "1",
      carClassVic %in% "03" ~ "2", 
      carClassVic %in% c("04", "06") ~ "3", 
      carClassVic %in% "05" ~ "4",
      carClassVic %in% c("07", "08", "09") ~ "5",
      carClassVic %in% c("10", "11") ~ "6",
      carClassVic %in% "12" ~ "7",
      carClassVic %in% c("98", "99", "Z1", "ZL", "##") ~ "8", # '##'은 없음(기물파손으로 추정)
      TRUE ~ NA
    )
  )

TA2$carClassVic <- factor(TA2$carClassVic)
summary(TA2$carClassVic)


#road_form_class
summary(TA2$road_form_class)

TA2 <- TA2 %>%
  mutate(road_form_class = as.character(road_form_class)) %>%
  mutate(
    road_form_class = case_when(
      road_form_class %in% "01" ~ "0", 
      road_form_class %in% "02" ~ "1",
      road_form_class %in% "05" ~ "2", 
      road_form_class %in% c("03", "04", "99", "Z3", "##") ~ "3",
      TRUE ~ NA
    )
  )

TA2$road_form_class <- factor(TA2$road_form_class)
summary(TA2$road_form_class)


#road_formD
summary(TA2$road_formD)

TA2 <- TA2 %>%
  mutate(road_formD = as.character(road_formD)) %>%
  mutate(
    road_formD = case_when(
      road_formD %in% "01" ~ "0", 
      road_formD %in% "02" ~ "1",
      road_formD %in% "04" ~ "2",
      road_formD %in% "05" ~ "3",
      road_formD %in% c("Z1", "Z2", "07") ~ "4",
      road_formD %in% c("06", "08") ~ "5",
      road_formD %in% c("03", "09", "10", "98", "99", "Z3", "##") ~ "6",
      TRUE ~ NA
    )
  )

TA2$road_formD <- factor(TA2$road_formD)
summary(TA2$road_formD)


#hour
summary(TA2$hour)

TA2 <- TA2 %>%
  mutate(hour = as.numeric(hour)) %>%
  mutate(
    hour = ifelse(hour >= 5 & hour < 11, 0, 
                  ifelse(hour >= 11 & hour < 17, 1,
                         ifelse(hour >= 17 & hour < 23, 2, 3)))
  )
  

TA2$hour <- factor(TA2$hour)
summary(TA2$hour)

# factor형 독립변수로 변경 --------------------------------------------------------

TA2$is_holiday <- factor(TA2$is_holiday)

# data selection ----------------------------------------------------------

total_TA <- TA2 %>%
  select(death_cnt, 
         fse_cnt, msl_cnt,
         accType, accTypeD, 
         carFLg, 
         carClassF, carClassVic, 
         road_form_class, road_formD, 
         lo_crd, la_crd,
         hour, date,
         is_holiday,
         sigungu
         )

# csv 저장 ------------------------------------------------------------------

# 데이터 저장
write.csv(total_TA, "TA_cleaned.csv", row.names = FALSE)

# 파일이 저장된 위치 확인
getwd()

# 지도 시각화 ------------------------------------------------------------------

# 위도와 경도 데이터 추출
map_data <- total_TA %>%
  select(lo_crd, la_crd, sigungu)

# 시군구 코드와 이름 맵핑
map_data <- map_data %>%
  mutate(
    sigungu_name = case_when(
      sigungu == 1401 ~ "춘천시",
      sigungu == 1402 ~ "원주시",
      sigungu == 1403 ~ "동해시",
      sigungu == 1404 ~ "강릉시",
      sigungu == 1405 ~ "속초시",
      sigungu == 1406 ~ "태백시",
      sigungu == 1407 ~ "삼척시",
      sigungu == 1412 ~ "홍천군",
      sigungu == 1413 ~ "횡성군",
      sigungu == 1415 ~ "영월군",
      sigungu == 1416 ~ "평창군",
      sigungu == 1417 ~ "정선군",
      sigungu == 1418 ~ "철원군",
      sigungu == 1419 ~ "화천군",
      sigungu == 1420 ~ "양구군",
      sigungu == 1421 ~ "인제군",
      sigungu == 1422 ~ "고성군",
      sigungu == 1423 ~ "양양군",
      TRUE ~ "기타"  # 기타로 처리 (예외값)
    )
  )

# 색상 팔레트 생성 (시군구 이름 기준)
palette <- colorFactor(
  palette = brewer.pal(n = min(9, length(unique(map_data$sigungu_name))), "Set1"),
  domain = map_data$sigungu_name
)

# 지도 생성
leaflet(data = map_data) %>%
  addTiles() %>%  # 기본 지도 타일 추가
  addCircleMarkers(
    lng = ~lo_crd, lat = ~la_crd,
    color = ~palette(sigungu_name),  # 시군구 이름에 따라 색상 지정
    radius = 4,                      # 마커 크기
    popup = ~paste("경도:", lo_crd, "<br>위도:", la_crd, "<br>시군구:", sigungu_name)
  ) %>%
  addLegend(
    "bottomright",              # 범례 위치
    pal = palette,              # 팔레트 설정
    values = ~sigungu_name,     # 시군구 이름을 기준으로 범례 생성
    title = "시군구별 색상",
    opacity = 1
  ) %>%
  setView(
    lng = mean(map_data$lo_crd, na.rm = TRUE), 
    lat = mean(map_data$la_crd, na.rm = TRUE), 
    zoom = 12
  )

