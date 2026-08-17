# Web Upscaler v2

SOOP과 YouTube의 HTML5 플레이어를 감지하고 WebGPU로 실시간 temporal 업스케일링하는 Chrome Manifest V3 확장입니다.

현재 구현된 처리 경로:

```text
HTMLVideoElement
  → copyExternalImageToTexture (프레임당 1회)
  → 입력 해상도 wide YCoCg deblock (4/8/16 px 경계, chroma 우선 완화)
  → 1/4 해상도 feature 분석
  → current patch 캐시 기반 diamond motion 추정 (속도/가속도 예측)
  → 원본 luma 기반 0.25/0.125 px motion refinement
  → LR 관측 표본을 2× HR phase lattice에 배치
  → RGBA16F premultiplied observation moments (Σw, Σwx, Σwx², Σw²) 재투영
  → variance와 effective sample count로 temporal 신뢰도 계산
  → observation과 분리된 latent HR radiance seed 생성
  → 이전 프레임의 최종 latent를 subpixel motion으로 재투영해 미관측 phase 초기화
  → latent HR을 LR pixel response로 재투영해 residual/sumSq 계산
  → 관측 phase constraint를 유지하며 robust YCoCg residual correction 2회
  → 화면 출력에서만 monotonic bicubic/edge-directed fallback과 temporal radiance 결합
  → 정확한 2× 내부 결과를 실제 플레이어 크기로 조건부 16-tap Lanczos2 resolve
  → 선택적 light sharpen
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
5. SOOP 방송 또는 YouTube 영상 페이지를 새로고침한 뒤 확장 팝업에서 활성화합니다.

코드를 다시 빌드했다면 `chrome://extensions`에서 확장을 먼저 새로고침하고,
그 다음 SOOP/YouTube 영상 탭도 새로고침해야 새 content script가 주입됩니다.

## 개발

```bash
npm run dev
```

- 팝업 미리보기: `http://127.0.0.1:5173/src/popup/index.html`
- Synthetic GPU harness: `http://127.0.0.1:5173/src/demo/index.html`
- 팝업과 harness의 `복원 진단`으로 coverage, variance, effective samples,
  residual 전/후, correction, motion, reactive mask 상태 확인
- 타입 검사: `npm run typecheck`
- WGSL 정적 검사: `npm run validate:shaders`
- 프로덕션 빌드: `npm run build`

## 구조

```text
src/
├─ background.ts             badge와 기본 설정
├─ content.ts                지원 사이트 페이지 진입점
├─ sites/                    SOOP/YouTube 플레이어 어댑터
├─ runtime/                  상태, 오버레이, 프레임 스케줄링
├─ gpu/                      WebGPU 리소스, 파이프라인, WGSL
├─ popup/                    설정 및 상태 UI
├─ demo/                     synthetic video 개발 harness
└─ shared/                   설정, 메시지, 상태 타입
```

## 현재 범위

이번 버전은 SOOP `video#livePlayer`와 YouTube HTML5 플레이어를 지원하며, YouTube의 SPA 탐색과 플레이어 교체 시 처리 세션을 다시 연결합니다. 가속도 예측 motion estimation, LR 관측 위치에 기반한 premultiplied HR observation moments 누적, 탐색·시간 불연속 시 history reset을 구현합니다. 정확한 2× 출력에서는 현재 LR 표본과 일치하는 even/even phase만 hard coverage로 인정하며, 공간 보간값은 observation history에 저장하지 않습니다. 별도 latent HR state는 관측 평균·분산·유효 표본 수와 motion-warp된 이전 최종 latent로 초기화합니다. 실제 관측 phase는 observation mean으로 반복 고정하고 미관측 phase에는 더 강한 LR residual back-projection을 적용하므로, 프레임이 이어질수록 두 번의 IBP가 누적 최적화처럼 수렴합니다. 540p 압축 노이즈에 맞춘 branchless reactive mask는 clamp 전 YCoCg history 차이와 motion error를 이용해 바뀐 자막·오버레이의 history, persistent latent, IBP, ringing resolve 기여를 즉시 낮춥니다. 입력 deblock pass는 실제 경계를 range weight로 보호하면서 4/8/16 px 코덱 경계와 chroma 얼룩을 temporal 누적 전에 완화합니다. GPU submit은 최대 2개만 pending으로 유지하며 GPU queue 시간은 저빈도로 표본 측정합니다. Auto 모드의 장치별 자동 튜닝과 실제 방송별 화질·성능 프로파일링은 후속 범위입니다.
