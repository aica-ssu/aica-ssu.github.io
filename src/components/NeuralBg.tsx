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
    const NODE_COUNT = 50;
    const CONNECT_DIST = 160;

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
        radius: Math.random() * 1.5 + 1,
      });
    }

    // Random active zone that drifts around
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
      const sparkRGB = dark ? [147, 197, 253] : [96, 165, 250];
      time += 0.016;

      // Drift the active zone smoothly
      zone.x += (zone.targetX - zone.x) * 0.008;
      zone.y += (zone.targetY - zone.y) * 0.008;
      if (time > nextTarget) {
        pickTarget();
        nextTarget = time + 2.5 + Math.random() * 2;
      }

      // Update node positions
      for (const node of nodes) {
        node.x += node.vx;
        node.y += node.vy;
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
      }

      // Draw connections near the active zone
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > CONNECT_DIST) continue;

          const midX = (nodes[i].x + nodes[j].x) / 2;
          const midY = (nodes[i].y + nodes[j].y) / 2;
          const zoneDist = Math.sqrt((midX - zone.x) ** 2 + (midY - zone.y) ** 2);
          const influence = Math.max(0, 1 - zoneDist / zone.radius);

          if (influence < 0.01) continue;

          const proximityAlpha = 1 - dist / CONNECT_DIST;
          const pulse = 0.7 + 0.3 * Math.sin(time * 5 + dist * 0.05);
          const alpha = influence * influence * proximityAlpha * pulse * 0.8;
          const [r, g, b] = accentRGB;

          const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
          grad.addColorStop(0, `rgba(${sparkRGB[0]},${sparkRGB[1]},255,${alpha * 0.7})`);
          grad.addColorStop(0.5, `rgba(${r},${g},${b},${alpha})`);
          grad.addColorStop(1, `rgba(${sparkRGB[0]},${sparkRGB[1]},255,${alpha * 0.7})`);

          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = grad;
          ctx.lineWidth = 0.5 + influence * 2.5;
          ctx.stroke();

          // Sparks
          if (influence > 0.2 && Math.random() < influence * 0.2) {
            const t = Math.random();
            const sx = nodes[i].x + (nodes[j].x - nodes[i].x) * t + (Math.random() - 0.5) * 12;
            const sy = nodes[i].y + (nodes[j].y - nodes[i].y) * t + (Math.random() - 0.5) * 12;
            ctx.beginPath();
            ctx.arc(sx, sy, Math.random() * 2.5 + 0.5, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${sparkRGB[0]},${sparkRGB[1]},255,${0.4 + Math.random() * 0.6})`;
            ctx.fill();
          }
        }
      }

      // Draw nodes
      for (const node of nodes) {
        const zoneDist = Math.sqrt((node.x - zone.x) ** 2 + (node.y - zone.y) ** 2);
        const influence = Math.max(0, 1 - zoneDist / zone.radius);
        const [r, g, b] = accentRGB;
        const alpha = 0.12 + influence * 0.7;
        const radius = node.radius + influence * 2;

        if (influence > 0.1) {
          const glowR = radius + 5 * influence;
          const glow = ctx.createRadialGradient(node.x, node.y, radius * 0.3, node.x, node.y, glowR);
          glow.addColorStop(0, `rgba(${sparkRGB[0]},${sparkRGB[1]},255,${influence * 0.2})`);
          glow.addColorStop(1, "rgba(0,0,0,0)");
          ctx.beginPath();
          ctx.arc(node.x, node.y, glowR, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();
        }

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
