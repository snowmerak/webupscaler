# Web Upscaler v2

SOOP과 YouTube의 HTML5 영상 위에서 동작하는 Chrome Manifest V3 WebGPU 업스케일러입니다.

현재 화상 처리 경로는 의도적으로 세 단계만 사용합니다.

```text
HTMLVideoElement
  → 정확한 가로 2× · 세로 2× HR 격자 생성
  → 프레임 간 움직임을 정렬해 실제 LR 관측이 도달한 서브픽셀만 누적
  → 복원된 2× 격자를 Lanczos2로 실제 캔버스 크기에 한 번 리사이징
  → WebGPU canvas
```

특징 분석과 모션 추정은 두 번째 단계의 정렬 계산입니다. 디블록, 샤프닝,
대비 곡선, 문자 보정, residual back-projection은 현재 출력 경로에 없습니다.

2× 격자의 네 칸 가운데 현재 프레임이 직접 관측한 phase만 observation
history에 기록합니다. 공간 보간값은 history에 넣지 않습니다. 앞뒤 프레임의
서브픽셀 이동이 다른 phase에 실제 관측을 제공하면, 관측 수와 분산에 따라
Lanczos 공간값을 시간축 관측 평균으로 교체합니다.

## 설치

```bash
npm install
npm run build
```

1. Chrome에서 `chrome://extensions`를 엽니다.
2. **개발자 모드**를 켭니다.
3. **압축해제된 확장 프로그램을 로드합니다**를 선택합니다.
4. 이 프로젝트의 `dist` 폴더를 선택합니다.
5. SOOP 또는 YouTube 영상 페이지를 새로고침하고 팝업에서 활성화합니다.

코드를 다시 빌드했다면 확장 프로그램과 영상 탭을 차례로 새로고침해야 합니다.

## 개발

```bash
npm run dev
```

- 팝업 미리보기: `http://127.0.0.1:5173/src/popup/index.html`
- Synthetic GPU harness: `http://127.0.0.1:5173/src/demo/index.html`
- 타입 검사: `npm run typecheck`
- WGSL 구조 검사: `npm run validate:shaders`
- 프로덕션 빌드: `npm run build`

`npm run dev`는 개발용 확장 파일을 만들 수 있으므로, Chrome에 로드하기 전에는
반드시 개발 서버를 종료하고 `npm run build`를 다시 실행합니다.

## 구조

```text
src/
├─ background.ts     배지와 기본 설정
├─ content.ts        지원 사이트 페이지 진입점
├─ sites/            SOOP/YouTube 플레이어 어댑터
├─ runtime/          상태, 오버레이, 프레임 스케줄링
├─ gpu/              WebGPU 리소스, 파이프라인, WGSL
├─ popup/            설정과 상태 UI
├─ demo/             synthetic video 개발 harness
└─ shared/           설정, 메시지, 상태 타입
```

영상 프레임 데이터는 브라우저 밖으로 전송되지 않습니다.
