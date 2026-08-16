# Web Upscaler v2

SOOP `video#livePlayer`를 감지하고 WebGPU로 실시간 공간 업스케일링하는 Chrome Manifest V3 확장입니다.

현재 구현된 처리 경로:

```text
HTMLVideoElement
  → copyExternalImageToTexture (프레임당 1회)
  → 1/4 해상도 feature 분석
  → 방향성 공간 복원
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
- 타입 검사: `npm run typecheck`
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

이번 버전은 SOOP-first 기반과 공간 업스케일 경로를 구현합니다. Temporal motion estimation과 history accumulation은 아직 포함하지 않았으며, 현재 런타임과 GPU 리소스 경계 위에 후속 구현할 수 있도록 분리돼 있습니다.

