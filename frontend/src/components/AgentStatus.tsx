import React, { useEffect, useState, useRef } from "react";
import { Terminal, Activity, Play, Clock } from "lucide-react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

interface LogEntry {
  message: string;
  type: "info" | "success" | "error";
  timestamp: string;
}

const AgentStatus: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isInitializing, setIsInitializing] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try {
      const res = await axios.get("http://localhost:8000/system/logs");
      setLogs(res.data);
    } catch (e) {
      console.error("Failed to fetch logs", e);
    }
  };

  useEffect(() => {
    fetchLogs();
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const handleInitialize = async () => {
    setIsInitializing(true);
    try {
      await axios.post("http://localhost:8000/system/initialize");
    } catch (e) {
      console.error("Init failed", e);
    }
    setTimeout(() => setIsInitializing(false), 2000);
  };

  return (
    <div className="card h-full flex flex-col min-h-[400px]">
      <div className="p-6 border-b border-border flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-lg">Agent Activity</h3>
          <p className="text-sm text-muted-foreground">Live system events</p>
        </div>
        <button
          onClick={handleInitialize}
          disabled={isInitializing}
          className="btn-secondary h-8 px-3 text-xs gap-2"
        >
          {isInitializing ? (
            <Activity size={14} className="animate-spin" />
          ) : (
            <Play size={14} />
          )}
          Initialize
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6">
        <AnimatePresence initial={false}>
          {logs.map((log, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative pl-6 border-l border-border last:border-0 pb-6 last:pb-0"
            >
              <div
                className={`absolute -left-[5px] top-1 w-2.5 h-2.5 rounded-full ring-4 ring-background ${
                  log.type === "error"
                    ? "bg-destructive"
                    : log.type === "success"
                    ? "bg-emerald-500"
                    : "bg-primary"
                }`}
              />

              <div className="flex flex-col gap-1">
                <p className="text-sm font-medium text-foreground leading-none">
                  {log.message}
                </p>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock size={12} />
                  <span>{new Date().toLocaleTimeString()}</span>
                </div>
              </div>
            </motion.div>
          ))}
          {logs.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground space-y-3">
              <div className="w-12 h-12 rounded-full bg-secondary flex items-center justify-center">
                <Terminal size={20} />
              </div>
              <p className="text-sm">No activity recorded</p>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default AgentStatus;
