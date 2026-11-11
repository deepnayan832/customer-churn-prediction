/**
 * 3D Customer Retention & Churn Dynamics Scene
 * Pure WebGL / Canvas implementation with zero external dependencies.
 * Fully compatible with strict Content-Security-Policy.
 */

export class RetentionOrbScene {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D | null = null;
  private animId: number | null = null;
  private isVisible = true;
  private targetRotX = 0.25;
  private targetRotY = 0.35;
  private currentRotX = 0.25;
  private currentRotY = 0.35;
  private churnRate = 0.265;
  private observer: IntersectionObserver | null = null;
  private particles: Array<{
    x: number;
    y: number;
    z: number;
    radius: number;
    color: string;
    speed: number;
    angle: number;
    orbitRadius: number;
    orbitSpeed: number;
    isChurn: boolean;
  }> = [];

  constructor(container: HTMLElement, initialChurnRate = 0.265) {
    this.churnRate = initialChurnRate;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "retention-3d-canvas";
    this.canvas.setAttribute("aria-hidden", "true");
    container.appendChild(this.canvas);

    this.ctx = this.canvas.getContext("2d", { alpha: true });
    this.initParticles();
    this.setupListeners();
    this.resize();
    this.start();
  }

  public setChurnRate(rate: number): void {
    this.churnRate = Math.max(0.01, Math.min(0.99, rate));
    this.initParticles();
  }

  private initParticles(): void {
    this.particles = [];
    const count = 140;
    const churnCount = Math.round(count * this.churnRate);

    for (let i = 0; i < count; i++) {
      const isChurn = i < churnCount;
      const angle = (i / count) * Math.PI * 2 + Math.random() * 0.4;
      const orbitRadius = isChurn
        ? 110 + Math.random() * 65
        : 65 + Math.random() * 45;

      this.particles.push({
        x: 0,
        y: 0,
        z: 0,
        radius: isChurn ? 1.8 + Math.random() * 2.2 : 2.0 + Math.random() * 2.5,
        color: isChurn ? "rgba(232, 109, 98, 0.85)" : "rgba(8, 145, 137, 0.9)",
        speed: (Math.random() * 0.008 + 0.004) * (isChurn ? -1 : 1),
        angle,
        orbitRadius,
        orbitSpeed: isChurn ? 0.012 : 0.008,
        isChurn,
      });
    }
  }

  private setupListeners(): void {
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (prefersReducedMotion) {
      this.isVisible = false;
      return;
    }

    window.addEventListener("resize", this.resize);

    const onPointerMove = (e: MouseEvent | TouchEvent) => {
      let clientX = 0;
      let clientY = 0;
      if ("touches" in e) {
        const touch = e.touches[0];
        if (!touch) return;
        clientX = touch.clientX;
        clientY = touch.clientY;
      } else {
        clientX = e.clientX;
        clientY = e.clientY;
      }

      const rect = this.canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;

      const normX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const normY = ((clientY - rect.top) / rect.height) * 2 - 1;

      this.targetRotY = normX * 0.6;
      this.targetRotX = -normY * 0.4;
    };

    window.addEventListener("mousemove", onPointerMove, { passive: true });
    window.addEventListener("touchmove", onPointerMove, { passive: true });

    if ("IntersectionObserver" in window) {
      this.observer = new IntersectionObserver((entries) => {
        const entry = entries[0];
        if (!entry) return;
        this.isVisible = entry.isIntersecting;
        if (this.isVisible && !this.animId) {
          this.start();
        }
      }, { threshold: 0.1 });
      this.observer.observe(this.canvas);
    }
  }

  private resize = (): void => {
    const parent = this.canvas.parentElement;
    if (!parent) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = parent.clientWidth || 360;
    const height = parent.clientHeight || 320;

    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    if (this.ctx) {
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
      this.ctx.scale(dpr, dpr);
    }
  };

  private start(): void {
    let lastTime = performance.now();

    const render = (time: number) => {
      if (!this.isVisible) {
        this.animId = null;
        return;
      }

      const dt = Math.min((time - lastTime) / 1000, 0.1);
      lastTime = time;

      this.update(dt);
      this.draw();

      this.animId = requestAnimationFrame(render);
    };

    this.animId = requestAnimationFrame(render);
  }

  private update(dt: number): void {
    this.currentRotX += (this.targetRotX - this.currentRotX) * 0.08;
    this.currentRotY += (this.targetRotY - this.currentRotY) * 0.08;

    this.currentRotY += dt * 0.15;

    for (const p of this.particles) {
      p.angle += p.orbitSpeed;
      const cosA = Math.cos(p.angle);
      const sinA = Math.sin(p.angle);
      const inclination = p.isChurn ? 0.45 : -0.25;

      p.x = cosA * p.orbitRadius;
      p.y = sinA * p.orbitRadius * Math.cos(inclination);
      p.z = sinA * p.orbitRadius * Math.sin(inclination);
    }
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx) return;

    const width = parseFloat(this.canvas.style.width) || 360;
    const height = parseFloat(this.canvas.style.height) || 320;
    const cx = width / 2;
    const cy = height / 2;

    ctx.clearRect(0, 0, width, height);

    const cosY = Math.cos(this.currentRotY);
    const sinY = Math.sin(this.currentRotY);
    const cosX = Math.cos(this.currentRotX);
    const sinX = Math.sin(this.currentRotX);

    const coreGrad = ctx.createRadialGradient(cx - 15, cy - 18, 4, cx, cy, 54);
    coreGrad.addColorStop(0, "rgba(227, 245, 242, 0.95)");
    coreGrad.addColorStop(0.3, "rgba(8, 145, 137, 0.85)");
    coreGrad.addColorStop(0.7, "rgba(6, 94, 89, 0.45)");
    coreGrad.addColorStop(1, "rgba(16, 38, 62, 0)");

    ctx.save();
    ctx.fillStyle = coreGrad;
    ctx.shadowColor = "rgba(8, 145, 137, 0.5)";
    ctx.shadowBlur = 32;
    ctx.beginPath();
    ctx.arc(cx, cy, 48, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    this.drawOrbitalRings(ctx, cx, cy, cosX, sinX, cosY, sinY);

    const fov = 340;
    const projected = this.particles.map((p) => {
      const x1 = p.x * cosY + p.z * sinY;
      const y1 = p.y;
      const z1 = -p.x * sinY + p.z * cosY;

      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      const scale = fov / (fov + z2 + 180);
      return {
        px: cx + x1 * scale,
        py: cy + y2 * scale,
        scale,
        depth: z2,
        radius: p.radius * scale,
        color: p.color,
        isChurn: p.isChurn,
      };
    });

    projected.sort((a, b) => b.depth - a.depth);

    for (const pt of projected) {
      if (pt.scale <= 0) continue;

      ctx.save();
      ctx.beginPath();
      ctx.arc(pt.px, pt.py, Math.max(0.6, pt.radius), 0, Math.PI * 2);
      ctx.fillStyle = pt.color;
      if (pt.depth > 0) {
        ctx.shadowColor = pt.isChurn ? "rgba(232, 109, 98, 0.6)" : "rgba(8, 145, 137, 0.6)";
        ctx.shadowBlur = 8 * pt.scale;
      }
      ctx.fill();
      ctx.restore();
    }
  }

  private drawOrbitalRings(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    cosX: number,
    sinX: number,
    cosY: number,
    sinY: number
  ): void {
    ctx.save();
    ctx.strokeStyle = "rgba(232, 109, 98, 0.28)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    for (let theta = 0; theta <= Math.PI * 2; theta += 0.08) {
      const rx = Math.cos(theta) * 135;
      const rz = Math.sin(theta) * 135;
      const x1 = rx * cosY + rz * sinY;
      const y1 = Math.sin(theta) * 22;
      const z1 = -rx * sinY + rz * cosY;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;
      const scale = 340 / (340 + z2 + 180);
      const px = cx + x1 * scale;
      const py = cy + y2 * scale;
      if (theta === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(8, 145, 137, 0.42)";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let theta = 0; theta <= Math.PI * 2; theta += 0.08) {
      const rx = Math.cos(theta) * 88;
      const rz = Math.sin(theta) * 88;
      const x1 = rx * cosY + rz * sinY;
      const y1 = -Math.sin(theta) * 18;
      const z1 = -rx * sinY + rz * cosY;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;
      const scale = 340 / (340 + z2 + 180);
      const px = cx + x1 * scale;
      const py = cy + y2 * scale;
      if (theta === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }

  public destroy(): void {
    if (this.animId) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }
    window.removeEventListener("resize", this.resize);
    if (this.observer) {
      this.observer.disconnect();
      this.observer = null;
    }
    this.canvas.remove();
  }
}
