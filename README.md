# Web Upscaler v2

SOOP `video#livePlayer`를 감지하고 WebGPU로 실시간 temporal 업스케일링하는 Chrome Manifest V3 확장입니다.

현재 구현된 처리 경로:

```text
HTMLVideoElement
  → copyExternalImageToTexture (프레임당 1회)
  → 1/4 해상도 feature 분석
  → current patch 캐시 기반 diamond motion 추정 (속도/가속도 예측)
  → 원본 luma 기반 0.25/0.125 px motion refinement
  → LR 관측 표본을 2× HR phase lattice에 배치
  → RGBA16F premultiplied observation/coverage history 재투영
  → 화면 출력에서만 방향성 공간 fallback과 temporal radiance 결합
  → 선택적 5-tap light sharpen
  → WebGPU canvas
```

원본 해상도가 현재 표시 크기에 충분하거나 WebGPU를 사용할 수 없는 경우에는 원본 영상을 그대로 표시합니다. 영상 프레임이나 시청 정보는 외부로 전송하지 않습니다.

## 설치

```bash
npm install
npm run build
```

1. Chrome에서 `chrome://extensions`를 엽니다.
2. **개발자 모드**를 켭니다.
3. **압축해제된 확장 프로그램을 로드합니다**를 선택합니다.
4. 이 프로젝트의 `dist` 폴더를 선택합니다.
5. SOOP 방송 페이지를 새로고침한 뒤 확장 팝업에서 활성화합니다.

## 개발

```bash
npm run dev
```

- 팝업 미리보기: `http://127.0.0.1:5173/src/popup/index.html`
- Synthetic GPU harness: `http://127.0.0.1:5173/src/demo/index.html`
- 팝업과 harness의 `HR coverage heatmap`으로 실제 관측 phase 누적 확인
- 타입 검사: `npm run typecheck`
- WGSL 정적 검사: `npm run validate:shaders`
- 프로덕션 빌드: `npm run build`

## 구조

```text
src/
├─ background.ts             badge와 기본 설정
├─ content.ts                SOOP 페이지 진입점
├─ sites/                    #livePlayer 우선 사이트 어댑터
├─ runtime/                  상태, 오버레이, 프레임 스케줄링
├─ gpu/                      WebGPU 리소스, 파이프라인, WGSL
├─ popup/                    설정 및 상태 UI
├─ demo/                     synthetic video 개발 harness
└─ shared/                   설정, 메시지, 상태 타입
```

## 현재 범위

이번 버전은 SOOP-first 기반, 가속도 예측 motion estimation, LR 관측 위치에 기반한 premultiplied HR observation/coverage 누적, 장면 전환/탐색 시 history reset을 구현합니다. 정확한 2× 출력에서는 현재 LR 표본과 일치하는 even/even phase만 hard coverage로 인정하며, 공간 보간값은 history에 저장하지 않습니다. GPU submit은 최대 2개만 pending으로 유지하며 GPU queue 시간은 저빈도로 표본 측정합니다. Auto 모드의 장치별 자동 튜닝과 실제 방송별 화질·성능 프로파일링은 후속 범위입니다.
