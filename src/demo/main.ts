import { WebGpuUpscaler } from '../gpu/webgpu-upscaler'
import type { BaseUpscalerSettings } from '../shared/settings'
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
  debugOverlay: false,
}
const target = { kind: 'upscale', width: 960, height: 540, scale: 2 } as const
const upscaler = new WebGpuUpscaler(output, (message) => {
  status.textContent = `GPU 연결 끊김: ${message}`
  status.dataset.tone = 'error'
}, { allowFallbackAdapter: true })
let animationFrame = 0
let animationStartedAt = performance.now()
let running = true
let inFlight = false
let processed = 0

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
    processed += 1
    status.textContent = '실행 중'
    status.dataset.tone = 'running'
    metrics.textContent = `처리 ${processed.toLocaleString()} 프레임 · 완료 지연 ${result.completionMs.toFixed(1)} ms${result.historyReset ? ' · history reset' : ''}`
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
