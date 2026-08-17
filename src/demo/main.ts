import { WebGpuUpscaler } from '../gpu/webgpu-upscaler'
import type { BaseUpscalerSettings, DiagnosticView } from '../shared/settings'
import './style.css'

function required<T extends Element>(selector: string) {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Demo element is missing: ${selector}`)
  return element
}

const source = required<HTMLCanvasElement>('#source')
const output = required<HTMLCanvasElement>('#output')
const video = required<HTMLVideoElement>('#video')
const status = required<HTMLElement>('#status')
const metrics = required<HTMLElement>('#metrics')
const toggle = required<HTMLButtonElement>('#toggle')
const deblockToggle = required<HTMLButtonElement>('#deblock-toggle')
const coverageToggle = required<HTMLButtonElement>('#coverage-toggle')
function get2dContext(canvas: HTMLCanvasElement) {
  const result = canvas.getContext('2d')
  if (!result) throw new Error('2D canvas를 초기화하지 못했습니다.')
  return result
}

const context = get2dContext(source)

const settings: BaseUpscalerSettings = {
  enabled: true,
  mode: 'balanced',
  requestedScale: 2,
  sharpness: 0.12,
  deblockStrength: 0.3,
  debugOverlay: false,
  diagnosticView: 'off',
}
const target = {
  kind: 'upscale',
  width: 960,
  height: 540,
  scale: 2,
  presentationWidth: 720,
  presentationHeight: 405,
} as const
const upscaler = new WebGpuUpscaler(output, (message) => {
  status.textContent = `GPU 연결 끊김: ${message}`
  status.dataset.tone = 'error'
}, { allowFallbackAdapter: true })
let animationFrame = 0
let animationStartedAt = performance.now()
let running = true
let inFlight = false
let processed = 0
let lastGpuQueueMs: number | null = null
const deblockLevels = [0, 0.3, 0.5]
let deblockLevelIndex = 1
const diagnosticViews: Array<{ value: DiagnosticView; label: string }> = [
  { value: 'off', label: '복원 진단: 끄기' },
  { value: 'coverage', label: '복원 진단: Coverage' },
  { value: 'variance', label: '복원 진단: Variance' },
  { value: 'samples', label: '복원 진단: Samples' },
  { value: 'residual', label: '복원 진단: Residual' },
  { value: 'correction', label: '복원 진단: Correction' },
  { value: 'motion', label: '복원 진단: Motion' },
  { value: 'reactive', label: '복원 진단: Reactive' },
]
let diagnosticViewIndex = 0

function draw(now: number) {
  const elapsed = (now - animationStartedAt) / 1000
  const width = source.width
  const height = source.height
  const gradient = context.createLinearGradient(0, 0, width, height)
  gradient.addColorStop(0, '#171526')
  gradient.addColorStop(0.5, '#41388c')
  gradient.addColorStop(1, '#171526')
  context.fillStyle = gradient
  context.fillRect(0, 0, width, height)

  context.save()
  context.translate((elapsed * 24) % 32, 0)
  for (let y = -16; y < height + 16; y += 16) {
    for (let x = -32; x < width + 16; x += 16) {
      context.fillStyle = (x / 16 + y / 16) % 2 === 0
        ? 'rgba(255,255,255,.07)'
        : 'rgba(0,0,0,.07)'
      context.fillRect(x, y, 16, 16)
    }
  }
  context.restore()

  // Static, low-contrast 8px blocks make it easy to compare the dedicated
  // source-resolution deblock pass without relying on a particular stream.
  const stressX = width - 164
  const stressY = 94
  context.fillStyle = '#69647d'
  context.fillRect(stressX, stressY, 136, 112)
  for (let y = 0; y < 112; y += 8) {
    for (let x = 0; x < 136; x += 8) {
      const delta = ((x / 8 * 3 + y / 8 * 5) % 5 - 2) * 3
      context.fillStyle = `rgb(${105 + delta}, ${100 + delta}, ${125 + delta})`
      context.fillRect(stressX + x, stressY + y, 8, 8)
    }
  }
  context.strokeStyle = 'rgba(255,255,255,.9)'
  context.lineWidth = 2
  context.beginPath()
  context.moveTo(stressX + 12, stressY + 92)
  context.lineTo(stressX + 124, stressY + 18)
  context.stroke()
  context.fillStyle = 'rgba(255,255,255,.72)'
  context.font = '11px ui-monospace, monospace'
  context.fillText('8px BLOCK STRESS', stressX + 8, stressY + 16)

  const circleX = width * 0.5 + Math.sin(elapsed * 1.4) * width * 0.28
  const circleY = height * 0.5 + Math.cos(elapsed * 1.1) * height * 0.16
  context.beginPath()
  context.arc(circleX, circleY, 38, 0, Math.PI * 2)
  context.fillStyle = '#7d6fff'
  context.fill()
  context.lineWidth = 3
  context.strokeStyle = '#d7d2ff'
  context.stroke()

  context.fillStyle = '#fff'
  context.font = '700 30px system-ui'
  context.fillText('WEB UPSCALER', 30, 54)
  context.fillStyle = 'rgba(255,255,255,.65)'
  context.font = '14px ui-monospace, monospace'
  context.fillText('synthetic subpixel motion · WebGPU', 31, 78)

  // Abrupt, fixed-position overlay changes exercise the reactive mask path
  // without conflating it with camera motion.
  const overlayLabels = ['LIVE 100', 'LIVE 250', 'SCENE READY']
  const overlayLabel = overlayLabels[Math.floor(elapsed * 1.5) % overlayLabels.length]
  context.fillStyle = 'rgba(18, 14, 34, .92)'
  context.fillRect(width - 176, 26, 148, 42)
  context.strokeStyle = '#ff6faf'
  context.lineWidth = 2
  context.strokeRect(width - 176, 26, 148, 42)
  context.fillStyle = '#fff1f7'
  context.font = '700 16px ui-monospace, monospace'
  context.fillText(overlayLabel, width - 164, 52)

  if (running) animationFrame = requestAnimationFrame(draw)
}

async function processFrame() {
  video.requestVideoFrameCallback(() => void processFrame())
  if (inFlight || !running) return
  inFlight = true
  try {
    const result = await upscaler.process(video, target, settings, {
      mediaTime: video.currentTime,
      skippedFrames: 0,
    })
    if (result === null) return
    processed += 1
    if (result.gpuQueueMs !== null) lastGpuQueueMs = result.gpuQueueMs
    status.textContent = '실행 중'
    status.dataset.tone = 'running'
    const gpuTime = lastGpuQueueMs === null ? '측정 대기' : `${lastGpuQueueMs.toFixed(1)} ms`
    metrics.textContent = `내부 ${target.width}×${target.height} → 표시 ${target.presentationWidth}×${target.presentationHeight} · 처리 ${processed.toLocaleString()} 프레임 · GPU queue ${gpuTime} · pending ${result.pendingSubmissions}/2${result.historyReset ? ' · history reset' : ''}`
  } catch (error) {
    status.textContent = '실행 오류'
    status.dataset.tone = 'error'
    metrics.textContent = error instanceof Error ? error.message : String(error)
  } finally {
    inFlight = false
  }
}

toggle.addEventListener('click', () => {
  running = !running
  toggle.textContent = running ? '애니메이션 멈추기' : '애니메이션 계속하기'
  if (running) {
    animationStartedAt = performance.now()
    animationFrame = requestAnimationFrame(draw)
  } else {
    cancelAnimationFrame(animationFrame)
  }
})

deblockToggle.addEventListener('click', () => {
  deblockLevelIndex = (deblockLevelIndex + 1) % deblockLevels.length
  settings.deblockStrength = deblockLevels[deblockLevelIndex]
  deblockToggle.textContent = `Deblock ${Math.round(settings.deblockStrength * 100)}%`
  upscaler.resetHistory()
})

coverageToggle.addEventListener('click', () => {
  diagnosticViewIndex = (diagnosticViewIndex + 1) % diagnosticViews.length
  const diagnostic = diagnosticViews[diagnosticViewIndex]
  settings.diagnosticView = diagnostic.value
  coverageToggle.textContent = diagnostic.label
})

async function initialize() {
  animationFrame = requestAnimationFrame(draw)
  const stream = source.captureStream(30)
  video.srcObject = stream
  await video.play()
  video.requestVideoFrameCallback(() => void processFrame())
}

void initialize().catch((error: unknown) => {
  status.textContent = '초기화 실패'
  status.dataset.tone = 'error'
  metrics.textContent = error instanceof Error ? error.message : String(error)
})
