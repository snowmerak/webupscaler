import { defineManifest } from '@crxjs/vite-plugin'

export default defineManifest({
  manifest_version: 3,
  name: 'Web Upscaler',
  description: 'A ready-to-extend Chrome extension for improving web images.',
  version: '0.1.0',
  action: {
    default_popup: 'src/popup/index.html',
    default_title: 'Web Upscaler',
  },
  background: {
    service_worker: 'src/background.ts',
    type: 'module',
  },
  permissions: ['activeTab', 'storage'],
  content_scripts: [
    {
      matches: ['http://*/*', 'https://*/*'],
      js: ['src/content.ts'],
      run_at: 'document_idle',
    },
  ],
})

