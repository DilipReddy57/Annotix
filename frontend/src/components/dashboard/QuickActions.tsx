"use client";

import { motion } from "framer-motion";
import {
  Plus,
  FolderOpen,
  Settings,
  Sparkles,
  ArrowRight,
  LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface QuickAction {
  icon: LucideIcon;
  label: string;
  description: string;
  color: "violet" | "emerald" | "amber";
  onClick: () => void;
}

interface QuickActionsProps {
  onNavigate: (path: string) => void;
}

const QuickActions = ({ onNavigate }: QuickActionsProps) => {
  const actions: QuickAction[] = [
    {
      icon: Plus,
      label: "New Annotation",
      description: "Start annotating assets",
      color: "violet",
      onClick: () => onNavigate("/studio"),
    },
    {
      icon: FolderOpen,
      label: "Browse Projects",
      description: "View all your projects",
      color: "emerald",
      onClick: () => onNavigate("/studio"),
    },
    {
      icon: Settings,
      label: "Settings",
      description: "Configure workspace",
      color: "amber",
      onClick: () => onNavigate("/settings"),
    },
  ];

  const colorMap = {
    violet: {
      icon: "bg-violet-500/10 text-violet-400 group-hover:bg-violet-500 group-hover:text-white",
      hover: "hover:border-violet-500/20",
    },
    emerald: {
      icon: "bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500 group-hover:text-white",
      hover: "hover:border-emerald-500/20",
    },
    amber: {
      icon: "bg-amber-500/10 text-amber-400 group-hover:bg-amber-500 group-hover:text-white",
      hover: "hover:border-amber-500/20",
    },
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2 px-1">
        <Sparkles size={14} className="text-violet-400" />
        <h3 className="text-sm font-medium text-foreground">Quick Actions</h3>
      </div>

      {/* Actions */}
      <div className="space-y-2">
        {actions.map((action, idx) => {
          const Icon = action.icon;
          const colors = colorMap[action.color];

          return (
            <motion.button
              key={action.label}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{
                duration: 0.4,
                delay: 0.2 + idx * 0.1,
                ease: [0.16, 1, 0.3, 1],
              }}
              onClick={action.onClick}
              className={cn(
                "w-full flex items-center gap-3 p-3 rounded-xl",
                "bg-white/[0.02] border border-white/[0.04]",
                "transition-all duration-200 group text-left",
                colors.hover
              )}
            >
              <div
                className={cn(
                  "w-10 h-10 rounded-lg flex items-center justify-center transition-all duration-200",
                  colors.icon
                )}
              >
                <Icon size={18} strokeWidth={1.5} />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium text-foreground">
                  {action.label}
                </p>
                <p className="text-xs text-muted-foreground">
                  {action.description}
                </p>
              </div>
              <ArrowRight
                size={14}
                className="text-muted-foreground opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200"
              />
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};

export default QuickActions;
