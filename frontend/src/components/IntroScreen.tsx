"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface IntroScreenProps {
  onComplete: () => void;
}

// Matrix characters
const MATRIX_CHARS =
  "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

const IntroScreen = ({ onComplete }: IntroScreenProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showTitle, setShowTitle] = useState(false);
  const [showSkip, setShowSkip] = useState(false);
  const [isExiting, setIsExiting] = useState(false);

  // Check visit count
  useEffect(() => {
    const visitCount = parseInt(
      localStorage.getItem("cortex_visits") || "0",
      10
    );
    const newCount = visitCount + 1;
    localStorage.setItem("cortex_visits", newCount.toString());

    // 3rd+ visit: auto-skip
    if (newCount >= 3) {
      onComplete();
      return;
    }

    // 2nd visit: show skip button
    if (newCount >= 2) {
      setShowSkip(true);
    }

    // Show title after 1.5s
    const titleTimer = setTimeout(() => setShowTitle(true), 1500);

    // Auto-complete after 4s
    const completeTimer = setTimeout(() => {
      handleExit();
    }, 4000);

    return () => {
      clearTimeout(titleTimer);
      clearTimeout(completeTimer);
    };
  }, [onComplete]);

  // Matrix rain effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Initialize drops
    const fontSize = 16;
    const columns = Math.floor(canvas.width / fontSize);
    const drops: number[] = Array(columns).fill(1);

    // Animation
    const draw = () => {
      // Semi-transparent black to create trail effect
      ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Matrix green text
      ctx.fillStyle = "#00ff41";
      ctx.font = `${fontSize}px "JetBrains Mono", monospace`;

      for (let i = 0; i < drops.length; i++) {
        const char =
          MATRIX_CHARS[Math.floor(Math.random() * MATRIX_CHARS.length)];
        ctx.fillText(char, i * fontSize, drops[i] * fontSize);

        // Reset drop randomly after it goes off screen
        if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
          drops[i] = 0;
        }
        drops[i]++;
      }
    };

    const interval = setInterval(draw, 50);

    return () => {
      clearInterval(interval);
      window.removeEventListener("resize", resize);
    };
  }, []);

  const handleExit = useCallback(() => {
    setIsExiting(true);
    setTimeout(onComplete, 500);
  }, [onComplete]);

  const handleSkip = () => {
    handleExit();
  };

  return (
    <AnimatePresence>
      {!isExiting && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="fixed inset-0 z-[100] bg-black flex items-center justify-center"
        >
          {/* Matrix Rain Canvas */}
          <canvas ref={canvasRef} className="absolute inset-0 opacity-60" />

          {/* Central Title */}
          <AnimatePresence>
            {showTitle && (
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.1 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="relative z-10 text-center"
              >
                {/* Glitch Effect Container */}
                <div className="relative">
                  <h1 className="text-6xl md:text-8xl font-bold font-display tracking-tighter">
                    <span className="text-white">CORTEX</span>
                    <span className="text-primary">.AI</span>
                  </h1>

                  {/* Glitch layers */}
                  <h1
                    className="absolute inset-0 text-6xl md:text-8xl font-bold font-display tracking-tighter text-cyan-400 opacity-70"
                    style={{
                      clipPath: "inset(10% 0 60% 0)",
                      transform: "translate(-2px, 0)",
                    }}
                    aria-hidden="true"
                  >
                    <span>CORTEX</span>
                    <span>.AI</span>
                  </h1>
                  <h1
                    className="absolute inset-0 text-6xl md:text-8xl font-bold font-display tracking-tighter text-red-400 opacity-70"
                    style={{
                      clipPath: "inset(60% 0 10% 0)",
                      transform: "translate(2px, 0)",
                    }}
                    aria-hidden="true"
                  >
                    <span>CORTEX</span>
                    <span>.AI</span>
                  </h1>
                </div>

                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                  className="mt-4 text-muted-foreground text-lg font-mono"
                >
                  Autonomous Annotation Platform
                </motion.p>

                {/* SAM3 Badge */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                  className="mt-3 flex items-center justify-center gap-2"
                >
                  <span className="px-3 py-1 text-xs font-mono bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30 rounded-full text-purple-300">
                    Powered by SAM3
                  </span>
                  <span className="px-3 py-1 text-xs font-mono bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-full text-green-300">
                    270K+ Concepts
                  </span>
                </motion.div>

                {/* Creator Credit */}
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.8, duration: 0.5 }}
                  className="mt-6 text-xs text-muted-foreground/60 font-mono"
                >
                  Created by <span className="text-primary">Dilip Reddy</span> •
                  Built with Antigravity
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Skip Button (only on 2nd visit) */}
          {showSkip && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              onClick={handleSkip}
              className="absolute bottom-8 right-8 px-4 py-2 text-sm text-muted-foreground hover:text-white border border-white/10 hover:border-white/30 rounded-lg transition-all backdrop-blur-sm"
            >
              Skip Intro →
            </motion.button>
          )}

          {/* Loading indicator */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 4, ease: "linear" }}
            className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-primary via-accent to-primary"
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default IntroScreen;
