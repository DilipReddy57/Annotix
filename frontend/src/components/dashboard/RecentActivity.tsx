"use client";

import { motion } from "framer-motion";
import { Clock, Image as ImageIcon, Video, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface ActivityItem {
  id: string;
  type: "image" | "video";
  name: string;
  project: string;
  time: string;
}

interface RecentActivityProps {
  items: ActivityItem[];
  isLoading?: boolean;
}

const RecentActivity = ({ items, isLoading = false }: RecentActivityProps) => {
  const navigate = useNavigate();

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Clock size={14} className="text-muted-foreground" />
          <h3 className="text-sm font-medium text-foreground">
            Recent Activity
          </h3>
        </div>
        {items.length > 0 && (
          <button
            onClick={() => navigate("/studio")}
            className="text-xs text-muted-foreground hover:text-violet-400 transition-colors flex items-center gap-1"
          >
            View all <ArrowRight size={12} />
          </button>
        )}
      </div>

      {/* Activity list */}
      {items.length > 0 ? (
        <div className="rounded-xl border border-white/[0.04] divide-y divide-white/[0.04] overflow-hidden">
          {items.slice(0, 4).map((item, idx) => (
            <motion.div
              key={item.id || idx}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3, delay: 0.5 + idx * 0.05 }}
              onClick={() => navigate("/studio")}
              className="flex items-center gap-3 p-3 bg-white/[0.01] hover:bg-white/[0.03] transition-colors cursor-pointer"
            >
              <div className="w-9 h-9 rounded-lg bg-white/[0.04] flex items-center justify-center text-muted-foreground">
                {item.type === "video" ? (
                  <Video size={16} strokeWidth={1.5} />
                ) : (
                  <ImageIcon size={16} strokeWidth={1.5} />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {item.name}
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  {item.project}
                </p>
              </div>
              <span className="text-xs text-muted-foreground shrink-0">
                {item.time}
              </span>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-white/[0.04] bg-white/[0.01] p-8 text-center">
          <div className="w-12 h-12 rounded-xl bg-white/[0.04] flex items-center justify-center mx-auto mb-3">
            <ImageIcon size={20} className="text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground">
            {isLoading ? "Loading activity..." : "No recent activity"}
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Upload files to get started
          </p>
        </div>
      )}
    </motion.div>
  );
};

export default RecentActivity;
