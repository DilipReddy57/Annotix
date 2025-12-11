"use client";

import { useRef, useMemo } from "react";
import { motion } from "framer-motion";

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
  "⊟",
  "⊠",
  "⊕",
  "⊖",
  "⊗",
  "⊘",
  "⌘",
  "⌥",
  "⎔",
  "⏣",
  "⏢",
  "⏤",
  "⏥",
  "car",
  "dog",
  "cat",
  "box",
  "mask",
  "0.95",
  "0.87",
  "0.92",
  "0.99",
  "0.78",
  "SAM",
  "AI",
  "ML",
  "DL",
  "CV",
  "bbox",
  "segm",
  "poly",
  "mask",
  "label",
];

interface Drop {
  id: number;
  x: number;
  speed: number;
  delay: number;
  symbols: string[];
  opacity: number;
}

const AnnotationRain = ({ className = "" }: AnnotationRainProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Generate rain drops
  const drops = useMemo(() => {
    const dropList: Drop[] = [];
    const numDrops = 25; // Number of vertical streams

    for (let i = 0; i < numDrops; i++) {
      const symbolCount = 8 + Math.floor(Math.random() * 12);
      const symbols = Array.from(
        { length: symbolCount },
        () =>
          ANNOTATION_SYMBOLS[
            Math.floor(Math.random() * ANNOTATION_SYMBOLS.length)
          ]
      );

      dropList.push({
        id: i,
        x: (i / numDrops) * 100 + (Math.random() * 4 - 2), // Spread across width
        speed: 15 + Math.random() * 20, // 15-35 seconds
        delay: Math.random() * 10, // Stagger start
        symbols,
        opacity: 0.1 + Math.random() * 0.2, // 0.1-0.3 opacity
      });
    }

    return dropList;
  }, []);

  return (
    <div
      ref={containerRef}
      className={`fixed inset-0 overflow-hidden pointer-events-none ${className}`}
      style={{ zIndex: 0 }}
    >
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-background" />

      {drops.map((drop) => (
        <motion.div
          key={drop.id}
          className="absolute top-0 flex flex-col gap-1 font-mono text-xs"
          style={{
            left: `${drop.x}%`,
            opacity: drop.opacity,
          }}
          initial={{ y: "-100%" }}
          animate={{ y: "100vh" }}
          transition={{
            duration: drop.speed,
            delay: drop.delay,
            repeat: Infinity,
            ease: "linear",
          }}
        >
          {drop.symbols.map((symbol, idx) => (
            <span
              key={idx}
              className={`
                ${idx === 0 ? "text-primary brightness-150" : ""}
                ${idx === 1 ? "text-primary brightness-125" : ""}
                ${idx === 2 ? "text-primary/80" : ""}
                ${idx > 2 ? "text-primary/40" : ""}
                ${symbol.length > 2 ? "text-[8px]" : "text-sm"}
              `}
              style={{
                textShadow: idx < 3 ? "0 0 10px currentColor" : "none",
              }}
            >
              {symbol}
            </span>
          ))}
        </motion.div>
      ))}

      {/* Additional glow effects */}
      <div className="absolute top-1/4 left-1/3 w-64 h-64 bg-primary/5 rounded-full blur-3xl animate-pulse" />
      <div
        className="absolute bottom-1/3 right-1/4 w-48 h-48 bg-accent/5 rounded-full blur-3xl animate-pulse"
        style={{ animationDelay: "1s" }}
      />
    </div>
  );
};

export default AnnotationRain;
