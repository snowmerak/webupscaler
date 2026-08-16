# Web Upscaler Chrome Extension

Chrome Manifest V3, TypeScript, Vite, CRXJS로 구성된 확장 프로그램 프로젝트입니다.

## 시작하기

```bash
npm install
npm run dev
```

Chrome에서 `chrome://extensions`를 열고 **개발자 모드**를 켠 다음, **압축해제된 확장 프로그램을 로드합니다**를 눌러 생성된 `dist` 폴더를 선택하세요.

개발 중에는 Vite가 파일 변경을 감지합니다. Manifest나 백그라운드 코드 변경 후에는 확장 프로그램을 새로고침해야 할 수 있습니다.

## 명령어

- `npm run dev` — 개발 모드 및 HMR
- `npm run typecheck` — TypeScript 검사
- `npm run build` — 배포용 `dist` 빌드
- `npm run preview` — 빌드 결과 미리보기

## 구조

```text
manifest.config.ts      Manifest V3 설정
src/background.ts       백그라운드 서비스 워커
src/content.ts          웹 페이지에서 실행되는 콘텐츠 스크립트
src/popup/              확장 프로그램 팝업 UI
```

