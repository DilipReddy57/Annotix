"use client";

import { motion } from "framer-motion";
import { Cpu, Zap, Database, Brain } from "lucide-react";

interface SystemStatusProps {
  isLoading?: boolean;
}

const SystemStatus = ({ isLoading = false }: SystemStatusProps) => {
  const systems = [
    {
      name: "SAM3 Model",
      status: "ready",
      icon: Brain,
      detail: "270K+ concepts",
      color: "violet",
    },
    {
      name: "RAG Engine",
      status: "ready",
      icon: Database,
      detail: "Label consistency",
      color: "emerald",
    },
    {
      name: "LLM Agent",
      status: "ready",
      icon: Zap,
      detail: "Auto-prompts",
      color: "amber",
    },
    {
      name: "Active Learning",
      status: "ready",
      icon: Cpu,
      detail: "Smart sampling",
      color: "sky",
    },
  ];

  const colorMap: Record<string, string> = {
    violet: "text-violet-400 bg-violet-500/10",
    emerald: "text-emerald-400 bg-emerald-500/10",
    amber: "text-amber-400 bg-amber-500/10",
    sky: "text-sky-400 bg-sky-500/10",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="rounded-xl bg-gradient-to-br from-card/80 to-card/40 border border-white/[0.04] p-5"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
        <h3 className="text-sm font-medium text-foreground">System Status</h3>
      </div>

      {/* Systems grid */}
      <div className="grid grid-cols-2 gap-3">
        {systems.map((system, idx) => {
          const Icon = system.icon;
          const colors = colorMap[system.color];

          return (
            <motion.div
              key={system.name}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.4 + idx * 0.1 }}
              className="flex items-center gap-2.5 p-2.5 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
            >
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center ${colors}`}
              >
                <Icon size={16} strokeWidth={1.5} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-foreground truncate">
                  {system.name}
                </p>
                <p className="text-[10px] text-muted-foreground">
                  {system.detail}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Footer status line */}
      <div className="mt-4 pt-3 border-t border-white/[0.04] flex items-center justify-between text-xs">
        <span className="text-muted-foreground">All systems operational</span>
        <span className="text-emerald-400 font-medium">Ready</span>
      </div>
    </motion.div>
  );
};

export default SystemStatus;
