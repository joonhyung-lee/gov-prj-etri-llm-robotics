# gov-prj-etri-llm-robotics
## [Korea University - ETRI project]

* Developing robotic systems that consider human preferences
* Each requirement is described in more detail below.

### 시스템 운영 요구사항

<!-- * SMR-01: 소형디바이스에서 돌아가는 로봇 환경 구축
  * 소형디바이스에 작동하는 거대언어모델을 활용한 로봇-사람 환경 구축
    1. Python이 수행 가능해야함
    2. 환경에 대한 상세 기술 -->

<!-- | SMR ID | 기능 | 상세 내용 | 세부 사항 | -->
| ID | 명칭 | 정의 | 세부 내용 |
|:---|:---|:---|:---|
| SFR-01 | 소형디바이스에서 돌아가는 로봇 환경 구축 | 소형디바이스에 작동하는 거대언어모델을 활용한 로봇-사람 환경 구축 | <ul><li>Python이 수행 가능해야함</li><li>환경에 대한 상세 기술</li></ul> |


### 기능 요구사항

<!-- * SFR-01: 거대언어모델의 로봇 행동 추정
  * 소형디바이스에서 작동하는 거대언어모델을 활용하여 로봇의 행동을 예측
    1. Python 수행 가능
    2. 사람의 언어기반 명령을 로봇의 행동 시퀀스로 변환 모듈
    3. Prompt 제공 및 코드 제공
* SFR-02: 거대언어모델을 기반으로 한 사람-로봇 대화 기능
  * 소형디바이스에서 작동하는 거대언어모델을 활용하여 사람-로봇 대화
    1. Python 환경 기반
    2. 로봇이 사람에게 모르는 정보를 되물어 명령을 수행하기에 충분한 정보를 얻는 모듈 구축
    3. Prompt 및 코드 제공
    4. 편리한 대화를 위한 인터페이스 기능
* SFR-03: 언어기반의 로봇행동을 실제 로봇의 궤적으로 바꿔주는 기능구축
  * 언어로 기술된 로봇 행동을 실제 로봇 궤적으로 바꿔주는 기능 구축
    1. Python 환경 및 ROS 환경
    2. 언어로 기술된 로봇행동을 parsing하여 궤적으로 바꿔주는 기능 구축
    3. 코드 제공
* SFR-04: 사용자의 선호도를 체계적으로 수집하고 분석하기 위한 피드백 시스템 구축
  * 로봇 행동에 대한 선호도를 정의하고, 그에 따른 사용자의 선호도를 수집/피드백 시스템을 구축 함. 
    1. 선호도별로 로봇 행동에 대한 분류(categorize)
    2. 사용자의 선호도 수집 시스템 구축
    3. 수집된 선호도를 바탕으로 로봇 행동에 대한 개선 및 피드백 시스템 구축 
* SFR-05: 로봇과의 상호작용에서 사용자의 요구를 파악하고 능동적으로 반영할 수 있는 시스템 개발
  * 로봇과 사용자 간의 상호작용을 개선하기 위해 사용자의 요구를 파악하고 그 요구를 능동적으로 반영할 수 있는 시스템을 개발
    1. 언어로 기술된 사용자의 요구를 이해하고 로봇의 행동에 반영하는 시스템 개발
    2. 상황, 맥락을 파악하여 사용자의 요구를 능동적으로 예측하는 시스템 개발 -->
<!-- | ------ | ---- | ------- | ------- | -------- | -->
<!-- | 요구사항 고유번호 | 요구사항 명칭 | 정의 | 세부 내용 | 코드 | -->

| ID | 명칭 | 정의 | 세부 내용 | 코드 |
|:---|:---|:---|:---|:---|
| SFR-01 | 거대언어모델의 로봇 행동 추정 | 소형디바이스에서 작동하는 거대언어모델을 활용하여 로봇의 행동을 예측 | <ul><li>Python 수행 가능</li><li>사람의 언어기반 명령을 로봇의 행동 시퀀스로 변환 모듈</li><li>Prompt 제공 및 코드 제공</li></ul> | [CLARA Demo](https://github.com/jeongeun980906/LLM-Uncertainty-DEMO) |
| SFR-02 | 거대언어모델을 기반으로 한 사람-로봇 대화 기능 | 소형디바이스에서 작동하는 거대언어모델을 활용하여 사람-로봇 대화 | <ul><li>Python 환경 기반</li><li>로봇이 사람에게 모르는 정보를 되물어 명령을 수행하기에 충분한 정보를 얻는 모듈 구축</li><li>Prompt 및 코드 제공</li><li>편리한 대화를 위한 인터페이스 기능</li></ul> | [CLARA Demo](https://github.com/jeongeun980906/LLM-Uncertainty-DEMO) |
| SFR-03 | 언어기반의 로봇행동을 실제 로봇의 궤적으로 바꿔주는 기능구축 | 언어로 기술된 로봇 행동을 실제 로봇 궤적으로 바꿔주는 기능 구축 | <ul><li>Python 환경 및 ROS 환경</li><li>언어로 기술된 로봇행동을 parsing하여 궤적으로 바꿔주는 기능 구축</li><li>코드 제공</li></ul> | [Jupter Demo](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/blob/main/scripts/realworld/realworld-code-snippets.ipynb), [Python Demo](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/blob/main/scripts/realworld/realworld-final.py) |
| SFR-04 | 사용자의 선호도를 체계적으로 수집하고 분석하기 위한 피드백 시스템 구축 | 로봇 행동에 대한 선호도를 정의하고, 그에 따른 사용자의 선호도를 수집/피드백 시스템을 구축함. | <ul><li>선호도별로 로봇 행동에 대한 분류(categorize)</li><li>사용자의 선호도 수집 시스템 구축</li><li>수집된 선호도를 바탕으로 로봇 행동에 대한 개선 및 피드백 시스템 구축</li></ul> | [Dataset](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/tree/main/dataset) |
| SFR-05 | 로봇과의 상호작용에서 사용자의 요구를 파악하고 능동적으로 반영할 수 있는 시스템 개발 | 로봇과 사용자 간의 상호작용을 개선하기 위해 사용자의 요구를 파악하고 그 요구를 능동적으로 반영할 수 있는 시스템을 개발 | <ul><li>언어로 기술된 사용자의 요구를 이해하고 로봇의 행동에 반영하는 시스템 개발</li><li>상황, 맥락을 파악하여 사용자의 요구를 능동적으로 예측하는 시스템 개발</li></ul> | [User-Feedback Demo](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/tree/main/scripts/CoVR/covr-user-feedback.py) <br/> Single Interaction Demo: [1](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/blob/main/scripts/CoVR/block-chain-of-visual-residuals.ipynb), [2](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/blob/main/scripts/CoVR/household-chain-of-visual-residuals.ipynb), [3](https://github.com/joonhyung-lee/gov-prj-etri-llm-robotics/blob/main/scripts/CoVR/polygon-interaction-reasoning-chain-of-visual-residuals.ipynb)|

### 인터페이스 요구사항

| ID | 명칭 | 정의 | 세부 내용 |
|:---|:---|:---|:---|
| SIR-01 | 사람-로봇 대화 인터페이스 | 로봇이 사람에게 모르는 정보를 되물어 명령을 수행하기에 충분한 정보를 얻는 대화 인터페이스 구축 | <ul><li>사용자와 로봇이 대화를 통해 해당 명령을 잘 수행 할 수 있는 인터페이스를 제공해야 함</li><li>타이핑 형식의 대화 수단을 사용한 인터페이스</li></ul> |
| SIR-02 | 프로그램의 실행 결과와 시스템 성능을 실시간으로 모니터링하고 이를 시각화하여 효과적으로 제공하는 기능 | 로봇의 궤적과 수집된 선호도 특징을 모니터링하고 이를 시각화하여 사용자 인터페이스를 제공함 | <ul><li>실제 로봇의 예상 궤적을 시뮬레이션에서 시각화하여 사용자에게 제공함.</li><li>수집된 선호도 특징을 그래프로 표현하여 사용자에게 직관적으로 전달함.</li></ul> |

### 프로젝트 관리요구사항

| ID | 명칭 | 정의 | 세부 내용 |
|:---|:---|:---|:---|
| SMR-01 | 주간보고를 통한 프로젝트 관리 | 주간보고를 통해서 프로젝트의 진도 및 협의 진행 | <ul><li>주간 기록한 연구 개발 일지를 제출함.</li></ul> |
