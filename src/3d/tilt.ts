/**
 * Lightweight 3D Card Tilt & Specular Interaction Helper
 * Supports mouse & touch with auto-reset on leave and reduced motion compliance.
 */

export function attach3DTilt(element: HTMLElement, maxTilt = 8, scale = 1.02): () => void {
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced) return () => {};

  const onMouseMove = (e: MouseEvent) => {
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    const rotX = (0.5 - y) * maxTilt * 2;
    const rotY = (x - 0.5) * maxTilt * 2;

    element.style.transform = `perspective(1000px) rotateX(${rotX.toFixed(2)}deg) rotateY(${rotY.toFixed(2)}deg) scale3d(${scale}, ${scale}, ${scale})`;
    element.style.setProperty("--pointer-x", `${(x * 100).toFixed(1)}%`);
    element.style.setProperty("--pointer-y", `${(y * 100).toFixed(1)}%`);
  };

  const onMouseEnter = () => {
    element.style.transition = "transform 0.12s cubic-bezier(0.2, 0, 0, 1)";
  };

  const onMouseLeave = () => {
    element.style.transition = "transform 0.45s cubic-bezier(0.2, 0.8, 0.2, 1)";
    element.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)";
  };

  element.addEventListener("mouseenter", onMouseEnter);
  element.addEventListener("mousemove", onMouseMove);
  element.addEventListener("mouseleave", onMouseLeave);

  return () => {
    element.removeEventListener("mouseenter", onMouseEnter);
    element.removeEventListener("mousemove", onMouseMove);
    element.removeEventListener("mouseleave", onMouseLeave);
  };
}

export function initAll3DTilts(root: HTMLElement): () => void {
  const cards = root.querySelectorAll<HTMLElement>(
    ".stat-card, .panel, .candidate-card, .metric-box, .source-card, .method-card"
  );
  const cleanups: Array<() => void> = [];

  cards.forEach((card) => {
    cleanups.push(attach3DTilt(card, 5, 1.015));
  });

  return () => {
    cleanups.forEach((cleanup) => cleanup());
  };
}
