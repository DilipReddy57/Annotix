"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: string | number;
  trend?: string;
  color?: "violet" | "emerald" | "amber" | "rose" | "sky";
  onClick?: () => void;
  delay?: number;
}

const colorStyles = {
  violet: {
    icon: "bg-violet-500/10 text-violet-400",
    glow: "group-hover:shadow-[0_0_30px_rgba(139,92,246,0.15)]",
    border: "group-hover:border-violet-500/20",
  },
  emerald: {
    icon: "bg-emerald-500/10 text-emerald-400",
    glow: "group-hover:shadow-[0_0_30px_rgba(16,185,129,0.15)]",
    border: "group-hover:border-emerald-500/20",
  },
  amber: {
    icon: "bg-amber-500/10 text-amber-400",
    glow: "group-hover:shadow-[0_0_30px_rgba(245,158,11,0.15)]",
    border: "group-hover:border-amber-500/20",
  },
  rose: {
    icon: "bg-rose-500/10 text-rose-400",
    glow: "group-hover:shadow-[0_0_30px_rgba(236,72,153,0.15)]",
    border: "group-hover:border-rose-500/20",
  },
  sky: {
    icon: "bg-sky-500/10 text-sky-400",
    glow: "group-hover:shadow-[0_0_30px_rgba(14,165,233,0.15)]",
    border: "group-hover:border-sky-500/20",
  },
};

const StatCard = ({
  icon: Icon,
  label,
  value,
  trend,
  color = "violet",
  onClick,
  delay = 0,
}: StatCardProps) => {
  const styles = colorStyles[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.5,
        delay: delay * 0.1,
        ease: [0.16, 1, 0.3, 1],
      }}
      onClick={onClick}
      className={cn(
        "group relative p-5 rounded-xl cursor-pointer transition-all duration-300",
        "bg-gradient-to-br from-card/80 to-card/40",
        "border border-white/[0.04]",
        styles.glow,
        styles.border,
        onClick && "cursor-pointer"
      )}
    >
      {/* Subtle gradient overlay on hover */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

      <div className="relative z-10">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div
            className={cn(
              "w-11 h-11 rounded-lg flex items-center justify-center transition-transform duration-300 group-hover:scale-110",
              styles.icon
            )}
          >
            <Icon size={22} strokeWidth={1.5} />
          </div>
          {trend && (
            <span className="text-xs font-medium text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">
              {trend}
            </span>
          )}
        </div>

        {/* Value */}
        <p className="text-3xl font-bold font-display tracking-tight text-foreground mb-1">
          {value}
        </p>

        {/* Label */}
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
    </motion.div>
  );
};

export default StatCard;
