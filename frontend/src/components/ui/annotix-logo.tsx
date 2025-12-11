"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface AnnotixLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
}

const AnnotixLogo = ({
  className,
  size = "md",
  showText = true,
}: AnnotixLogoProps) => {
  const sizes = {
    sm: { icon: 28, text: "text-lg", gap: "gap-2" },
    md: { icon: 36, text: "text-xl", gap: "gap-2.5" },
    lg: { icon: 48, text: "text-3xl", gap: "gap-3" },
    xl: { icon: 64, text: "text-5xl", gap: "gap-4" },
  };

  const dim = sizes[size];

  return (
    <div className={cn("flex items-center", dim.gap, className)}>
      {/* SVG Logo Mark - Bounding Box with Target */}
      <motion.svg
        width={dim.icon}
        height={dim.icon}
        viewBox="0 0 48 48"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        whileHover={{ scale: 1.05, rotate: 5 }}
        transition={{ duration: 0.2 }}
        className="drop-shadow-[0_0_10px_rgba(16,185,129,0.4)]"
      >
        {/* Outer bounding box */}
        <rect
          x="4"
          y="4"
          width="40"
          height="40"
          rx="4"
          stroke="#10b981"
          strokeWidth="2.5"
          strokeDasharray="8 4"
          fill="none"
        />

        {/* Corner brackets - top left */}
        <path
          d="M4 14 L4 4 L14 4"
          stroke="#10b981"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
        />

        {/* Corner brackets - top right */}
        <path
          d="M34 4 L44 4 L44 14"
          stroke="#10b981"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
        />

        {/* Corner brackets - bottom left */}
        <path
          d="M4 34 L4 44 L14 44"
          stroke="#10b981"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
        />

        {/* Corner brackets - bottom right */}
        <path
          d="M34 44 L44 44 L44 34"
          stroke="#10b981"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
        />

        {/* Center crosshair */}
        <circle cx="24" cy="24" r="6" fill="#10b981" opacity="0.2" />
        <circle cx="24" cy="24" r="3" fill="#10b981" />

        {/* Crosshair lines */}
        <line
          x1="24"
          y1="14"
          x2="24"
          y2="20"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <line
          x1="24"
          y1="28"
          x2="24"
          y2="34"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <line
          x1="14"
          y1="24"
          x2="20"
          y2="24"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <line
          x1="28"
          y1="24"
          x2="34"
          y2="24"
          stroke="#10b981"
          strokeWidth="2"
          strokeLinecap="round"
        />
      </motion.svg>

      {/* Text Logo */}
      {showText && (
        <motion.span
          className={cn("font-bold font-display tracking-tight", dim.text)}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <span className="text-white">ANNOT</span>
          <span className="text-emerald-500">IX</span>
        </motion.span>
      )}
    </div>
  );
};

export default AnnotixLogo;
