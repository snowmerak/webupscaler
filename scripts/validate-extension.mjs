import { access, readFile, readdir } from 'node:fs/promises'
import { resolve } from 'node:path'

const outputRoot = resolve('dist')
const manifest = JSON.parse(await readFile(resolve(outputRoot, 'manifest.json'), 'utf8'))
const requiredFiles = [
  manifest.background?.service_worker,
  manifest.action?.default_popup,
  ...(manifest.content_scripts ?? []).flatMap((entry) => entry.js ?? []),
].filter(Boolean)

for (const file of requiredFiles) {
  await access(resolve(outputRoot, file))
}

async function collectFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const files = []
  for (const entry of entries) {
    const path = resolve(directory, entry.name)
    if (entry.isDirectory()) files.push(...await collectFiles(path))
    else files.push(path)
  }
  return files
}

const forbiddenDevelopmentMarkers = [
  'CRXJS DEV MODE',
  '@crx/client-worker',
  '@vite/env',
  'http://localhost:',
]

for (const file of await collectFiles(outputRoot)) {
  if (!/\.(?:html|js|json)$/.test(file)) continue
  const source = await readFile(file, 'utf8')
  const marker = forbiddenDevelopmentMarkers.find((candidate) => source.includes(candidate))
  if (marker) throw new Error(`${file}: development marker found: ${marker}`)

  for (const match of source.matchAll(/chrome\.runtime\.getURL\(["']([^"']+)["']\)/g)) {
    await access(resolve(outputRoot, match[1]))
  }
}

console.log(`✓ extension package: ${requiredFiles.length} manifest targets and dynamic imports verified`)
