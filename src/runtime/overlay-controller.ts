export class OverlayController {
  readonly canvas: HTMLCanvasElement
  private readonly debugElement: HTMLDivElement
  private readonly host: HTMLElement
  private readonly video: HTMLVideoElement
  private readonly previousHostPosition: string
  private readonly resizeObserver: ResizeObserver
  private layoutFrame: number | null = null

  constructor(video: HTMLVideoElement, host: HTMLElement) {
    this.video = video
    this.host = host
    this.previousHostPosition = host.style.position

    if (getComputedStyle(host).position === 'static') {
      host.style.position = 'relative'
    }

    this.canvas = document.createElement('canvas')
    this.canvas.dataset.webUpscaler = 'canvas'
    this.canvas.setAttribute('aria-hidden', 'true')
    Object.assign(this.canvas.style, {
      position: 'absolute',
      pointerEvents: 'none',
      zIndex: '2',
      display: 'none',
      objectFit: 'contain',
    })

    this.debugElement = document.createElement('div')
    this.debugElement.dataset.webUpscaler = 'debug'
    Object.assign(this.debugElement.style, {
      position: 'absolute',
      zIndex: '3',
      display: 'none',
      padding: '7px 9px',
      color: '#f8fafc',
      background: 'rgba(8, 10, 18, 0.78)',
      border: '1px solid rgba(255,255,255,0.16)',
      borderRadius: '7px',
      font: '11px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace',
      whiteSpace: 'pre',
      pointerEvents: 'none',
    })

    host.append(this.canvas, this.debugElement)
    this.resizeObserver = new ResizeObserver(() => this.scheduleLayout())
    this.resizeObserver.observe(video)
    this.resizeObserver.observe(host)
    window.addEventListener('resize', this.scheduleLayout, { passive: true })
    window.addEventListener('scroll', this.scheduleLayout, { passive: true, capture: true })
    document.addEventListener('fullscreenchange', this.scheduleLayout)
    this.layout()
  }

  private readonly scheduleLayout = () => {
    if (this.layoutFrame !== null) return
    this.layoutFrame = requestAnimationFrame(() => {
      this.layoutFrame = null
      this.layout()
    })
  }

  layout() {
    const videoRect = this.video.getBoundingClientRect()
    const hostRect = this.host.getBoundingClientRect()
    const left = videoRect.left - hostRect.left + this.host.scrollLeft
    const top = videoRect.top - hostRect.top + this.host.scrollTop

    Object.assign(this.canvas.style, {
      left: `${left}px`,
      top: `${top}px`,
      width: `${videoRect.width}px`,
      height: `${videoRect.height}px`,
      borderRadius: getComputedStyle(this.video).borderRadius,
    })
    Object.assign(this.debugElement.style, {
      left: `${left + 12}px`,
      top: `${top + 12}px`,
    })
  }

  show() {
    this.canvas.style.display = 'block'
  }

  hide() {
    this.canvas.style.display = 'none'
  }

  setDebug(text: string, visible: boolean) {
    this.debugElement.textContent = text
    this.debugElement.style.display = visible ? 'block' : 'none'
  }

  destroy() {
    this.resizeObserver.disconnect()
    window.removeEventListener('resize', this.scheduleLayout)
    window.removeEventListener('scroll', this.scheduleLayout, { capture: true })
    document.removeEventListener('fullscreenchange', this.scheduleLayout)
    if (this.layoutFrame !== null) cancelAnimationFrame(this.layoutFrame)
    this.canvas.remove()
    this.debugElement.remove()
    this.host.style.position = this.previousHostPosition
  }
}

