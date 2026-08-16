import { defineManifest } from '@crxjs/vite-plugin'

export default defineManifest({
  manifest_version: 3,
  name: 'Web Upscaler v2',
  description: 'SOOP 영상을 WebGPU로 실시간 업스케일링합니다.',
  version: '0.1.0',
  minimum_chrome_version: '121',
  action: {
    default_popup: 'src/popup/index.html',
    default_title: 'Web Upscaler',
  },
  background: {
    service_worker: 'src/background.ts',
    type: 'module',
  },
  permissions: ['storage'],
  content_scripts: [
    {
      matches: [
        'https://play.sooplive.com/*',
        'https://play.sooplive.co.kr/*',
      ],
      js: ['src/content.ts'],
      run_at: 'document_idle',
    },
  ],
})
