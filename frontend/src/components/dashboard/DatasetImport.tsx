"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Link2,
  Download,
  Loader2,
  CheckCircle,
  AlertCircle,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface DatasetImportProps {
  onImport: (url: string, source: string) => Promise<void>;
}

const DatasetImport = ({ onImport }: DatasetImportProps) => {
  const [url, setUrl] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [status, setStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [errorMessage, setErrorMessage] = useState("");

  const detectSource = (inputUrl: string): string => {
    if (inputUrl.includes("kaggle.com")) return "kaggle";
    if (inputUrl.includes("huggingface.co")) return "huggingface";
    if (inputUrl.includes("github.com")) return "github";
    if (inputUrl.includes("drive.google.com")) return "gdrive";
    return "url";
  };

  const getSourceLabel = (source: string): string => {
    const labels: Record<string, string> = {
      kaggle: "Kaggle Dataset",
      huggingface: "HuggingFace Dataset",
      github: "GitHub Repository",
      gdrive: "Google Drive",
      url: "Direct URL",
    };
    return labels[source] || "URL";
  };

  const handleImport = async () => {
    if (!url.trim()) return;

    setIsImporting(true);
    setStatus("loading");
    setErrorMessage("");

    try {
      const source = detectSource(url);
      await onImport(url.trim(), source);
      setStatus("success");
      setUrl("");
      setTimeout(() => setStatus("idle"), 3000);
    } catch (error: any) {
      setStatus("error");
      setErrorMessage(error.message || "Failed to import dataset");
      setTimeout(() => setStatus("idle"), 5000);
    } finally {
      setIsImporting(false);
    }
  };

  const source = url ? detectSource(url) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="mt-4"
    >
      {/* Section header */}
      <div className="flex items-center gap-2 mb-3">
        <Link2 size={14} className="text-primary" />
        <span className="text-sm font-medium text-foreground">
          Import from URL
        </span>
        <span className="text-xs text-muted-foreground">
          (Kaggle, HuggingFace, GitHub)
        </span>
      </div>

      {/* Input area */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Paste Kaggle, HuggingFace, or dataset URL..."
            disabled={isImporting}
            className={cn(
              "w-full px-4 py-3 rounded-xl text-sm",
              "bg-white/[0.03] border border-white/10",
              "text-foreground placeholder:text-muted-foreground",
              "focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/20",
              "transition-all duration-200",
              isImporting && "opacity-50 cursor-not-allowed"
            )}
            onKeyDown={(e) => e.key === "Enter" && handleImport()}
          />

          {/* Source badge */}
          <AnimatePresence>
            {source && url && (
              <motion.span
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className={cn(
                  "absolute right-3 top-1/2 -translate-y-1/2",
                  "text-[10px] font-medium px-2 py-0.5 rounded-full",
                  source === "kaggle" && "bg-blue-500/20 text-blue-400",
                  source === "huggingface" &&
                    "bg-yellow-500/20 text-yellow-400",
                  source === "github" && "bg-gray-500/20 text-gray-400",
                  source === "gdrive" && "bg-green-500/20 text-green-400",
                  source === "url" && "bg-primary/20 text-primary"
                )}
              >
                {getSourceLabel(source)}
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        {/* Import button */}
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleImport}
          disabled={!url.trim() || isImporting}
          className={cn(
            "px-5 py-3 rounded-xl font-medium text-sm flex items-center gap-2",
            "transition-all duration-200",
            url.trim() && !isImporting
              ? "bg-primary text-primary-foreground hover:bg-primary/90"
              : "bg-white/5 text-muted-foreground cursor-not-allowed"
          )}
        >
          {status === "loading" && (
            <Loader2 size={16} className="animate-spin" />
          )}
          {status === "success" && <CheckCircle size={16} />}
          {status === "error" && <AlertCircle size={16} />}
          {status === "idle" && <Download size={16} />}
          <span>
            {status === "loading"
              ? "Importing..."
              : status === "success"
              ? "Done!"
              : status === "error"
              ? "Failed"
              : "Import"}
          </span>
        </motion.button>
      </div>

      {/* Error message */}
      <AnimatePresence>
        {status === "error" && errorMessage && (
          <motion.p
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="text-xs text-red-400 mt-2"
          >
            {errorMessage}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Supported sources */}
      <div className="flex flex-wrap gap-2 mt-3">
        {[
          { name: "Kaggle", icon: "📊", url: "kaggle.com/datasets" },
          { name: "HuggingFace", icon: "🤗", url: "huggingface.co/datasets" },
          { name: "GitHub", icon: "🐙", url: "github.com" },
        ].map((source) => (
          <a
            key={source.name}
            href={`https://${source.url}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
          >
            <span>{source.icon}</span>
            <span>{source.name}</span>
            <ExternalLink size={8} />
          </a>
        ))}
      </div>
    </motion.div>
  );
};

export default DatasetImport;
