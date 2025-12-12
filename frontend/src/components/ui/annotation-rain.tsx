"use client";

import { useMemo } from "react";

interface AnnotationRainProps {
  className?: string;
}

// Annotation-themed symbols for the rain effect
const ANNOTATION_SYMBOLS = [
  "□",
  "■",
  "◇",
  "◆",
  "○",
  "●",
  "△",
  "▲",
  "▽",
  "▼",
  "⬡",
  "⬢",
  "⊡",
  "⊞",
  "⊕",
  "⊗",
  "⌘",
  "car",
  "dog",
  "cat",
  "mask",
  "0.95",
  "0.87",
  "0.92",
  "SAM",
  "AI",
  "ML",
  "bbox",
  "segm",
  "poly",
  "label",
];

interface Drop {
  id: number;
  x: number;
  duration: number;
  delay: number;
  symbols: string[];
  colorVar: string;
}

const AnnotationRain = ({ className = "" }: AnnotationRainProps) => {
  // Color palette - rich, artistic colors
  const colors = [
    "rgb(139, 92, 246)", // Violet
    "rgb(16, 185, 129)", // Emerald
    "rgb(59, 130, 246)", // Blue
    "rgb(236, 72, 153)", // Pink
    "rgb(245, 158, 11)", // Amber
  ];

  // Generate rain drops - memoized for performance
  const drops = useMemo(() => {
    const dropList: Drop[] = [];
    const numDrops = 18; // Optimized drop count

    for (let i = 0; i < numDrops; i++) {
      const symbolCount = 6 + Math.floor(Math.random() * 8);
      const symbols = Array.from(
        { length: symbolCount },
        () =>
          ANNOTATION_SYMBOLS[
            Math.floor(Math.random() * ANNOTATION_SYMBOLS.length)
          ]
      );

      dropList.push({
        id: i,
        x: (i / numDrops) * 100 + (Math.random() * 4 - 2),
        duration: 12 + Math.random() * 15, // 12-27 seconds
        delay: Math.random() * 8,
        symbols,
        colorVar: colors[i % colors.length],
      });
    }

    return dropList;
  }, []);

  return (
    <div
      className={`fixed inset-0 overflow-hidden pointer-events-none ${className}`}
      style={{ zIndex: 0 }}
    >
      {/* Animated gradient background */}
      <div
        className="absolute inset-0 opacity-30"
        style={{
          background: `
            radial-gradient(ellipse at 20% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(59, 130, 246, 0.08) 0%, transparent 60%)
          `,
        }}
      />

      {/* Rain drops - CSS animation only */}
      {drops.map((drop) => (
        <div
          key={drop.id}
          className="absolute top-0 flex flex-col gap-0.5 font-mono"
          style={{
            left: `${drop.x}%`,
            animation: `rainFall ${drop.duration}s linear ${drop.delay}s infinite`,
            willChange: "transform",
          }}
        >
          {drop.symbols.map((symbol, idx) => {
            const isHead = idx < 3;
            const opacity =
              idx === 0 ? 0.9 : idx === 1 ? 0.7 : idx === 2 ? 0.5 : 0.2;
            const fontSize = symbol.length > 2 ? "8px" : "11px";

            return (
              <span
                key={idx}
                className="leading-tight"
                style={{
                  color: drop.colorVar,
                  opacity,
                  fontSize,
                  textShadow: isHead ? `0 0 12px ${drop.colorVar}` : "none",
                  filter: isHead ? "brightness(1.3)" : "none",
                }}
              >
                {symbol}
              </span>
            );
          })}
        </div>
      ))}

      {/* Floating orbs - subtle ambient glow */}
      <div
        className="absolute w-96 h-96 rounded-full blur-3xl"
        style={{
          top: "15%",
          left: "10%",
          background:
            "radial-gradient(circle, rgba(139, 92, 246, 0.08) 0%, transparent 70%)",
          animation: "orbFloat 20s ease-in-out infinite",
        }}
      />
      <div
        className="absolute w-72 h-72 rounded-full blur-3xl"
        style={{
          bottom: "20%",
          right: "15%",
          background:
            "radial-gradient(circle, rgba(16, 185, 129, 0.06) 0%, transparent 70%)",
          animation: "orbFloat 25s ease-in-out infinite reverse",
        }}
      />
      <div
        className="absolute w-48 h-48 rounded-full blur-3xl"
        style={{
          top: "50%",
          right: "30%",
          background:
            "radial-gradient(circle, rgba(236, 72, 153, 0.05) 0%, transparent 70%)",
          animation: "orbFloat 18s ease-in-out 3s infinite",
        }}
      />

      {/* Gradient fade at bottom for content readability */}
      <div
        className="absolute bottom-0 left-0 right-0 h-64 pointer-events-none"
        style={{
          background:
            "linear-gradient(to top, var(--color-background) 0%, transparent 100%)",
        }}
      />
    </div>
  );
};

export default AnnotationRain;
