"use client";

import { useEffect, useRef } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

export default function NeuralBg() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    const nodes: Node[] = [];
    const isMobile = window.innerWidth < 768;
    const NODE_COUNT = isMobile ? 35 : 80;
    const CONNECT_DIST = isMobile ? 140 : 170;

    const resize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      canvas.width = parent.offsetWidth;
      canvas.height = parent.offsetHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.2 + 0.8,
      });
    }

    // Drifting active zone
    const zone = { x: 0, y: 0, targetX: 0, targetY: 0, radius: 280 };
    const pickTarget = () => {
      zone.targetX = Math.random() * canvas.width;
      zone.targetY = Math.random() * canvas.height;
    };
    zone.x = canvas.width / 2;
    zone.y = canvas.height / 2;
    pickTarget();

    const isDark = () => document.documentElement.classList.contains("dark");
    let time = 0;
    let nextTarget = 2;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const dark = isDark();
      const accentRGB = dark ? [96, 165, 250] : [37, 99, 235];
      time += 0.016;

      // Drift zone
      zone.x += (zone.targetX - zone.x) * 0.008;
      zone.y += (zone.targetY - zone.y) * 0.008;
      if (time > nextTarget) {
        pickTarget();
        nextTarget = time + 2.5 + Math.random() * 2;
      }

      // Update positions
      for (const node of nodes) {
        node.x += node.vx;
        node.y += node.vy;
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
      }

      const [r, g, b] = accentRGB;

      // Draw connections with directional glow
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > CONNECT_DIST) continue;

          const midX = (nodes[i].x + nodes[j].x) / 2;
          const midY = (nodes[i].y + nodes[j].y) / 2;
          const zoneDist = Math.sqrt((midX - zone.x) ** 2 + (midY - zone.y) ** 2);
          const influence = Math.max(0, 1 - zoneDist / zone.radius);

          if (influence < 0.01) continue;

          const proximityAlpha = 1 - dist / CONNECT_DIST;
          const baseAlpha = influence * influence * proximityAlpha * 0.7;

          // Animated pulse traveling along the line
          // Phase based on time + unique offset per pair
          const phase = (time * 1.2 + (i * 7 + j * 13) * 0.1) % 1;

          const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);

          // Build gradient: dim → bright peak → dim (peak moves along line)
          const peakPos = phase;
          const spread = 0.25;
          const dimAlpha = baseAlpha * 0.3;
          const brightAlpha = baseAlpha;

          grad.addColorStop(0, `rgba(${r},${g},${b},${dimAlpha})`);
          if (peakPos - spread > 0) {
            grad.addColorStop(Math.max(0, peakPos - spread), `rgba(${r},${g},${b},${dimAlpha})`);
          }
          grad.addColorStop(Math.max(0, Math.min(1, peakPos)), `rgba(${r},${g},${b},${brightAlpha})`);
          if (peakPos + spread < 1) {
            grad.addColorStop(Math.min(1, peakPos + spread), `rgba(${r},${g},${b},${dimAlpha})`);
          }
          grad.addColorStop(1, `rgba(${r},${g},${b},${dimAlpha})`);

          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.5 + influence * 1.5;
          ctx.stroke();
        }
      }

      // Draw nodes (subtle)
      for (const node of nodes) {
        const zoneDist = Math.sqrt((node.x - zone.x) ** 2 + (node.y - zone.y) ** 2);
        const influence = Math.max(0, 1 - zoneDist / zone.radius);
        const alpha = 0.1 + influence * 0.5;
        const radius = node.radius + influence * 1;

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fill();
      }

      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ zIndex: 0 }}
    />
  );
}
