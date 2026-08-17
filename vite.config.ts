import { crx } from '@crxjs/vite-plugin'
import { defineConfig } from 'vite'

import manifest from './manifest.config.ts'

export default defineConfig({
  plugins: [crx({ manifest })],
  build: {
    rollupOptions: {
      output: {
        // Keep extension entry URLs stable across local rebuilds. Chrome may
        // still have an earlier manifest/content-script loader in memory while
        // dist is replaced; hashed filenames make that loader disappear.
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
  },
})
