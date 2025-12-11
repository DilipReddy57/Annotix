"use client";

import { useRef, useCallback, useState } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  Image as ImageIcon,
  Video,
  Loader2,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface HeroUploadProps {
  onFilesSelected: (files: File[]) => void;
  isUploading?: boolean;
  uploadStatus?: "idle" | "uploading" | "success" | "error";
}

const HeroUpload = ({
  onFilesSelected,
  isUploading = false,
  uploadStatus = "idle",
}: HeroUploadProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files);
      onFilesSelected(files);
    },
    [onFilesSelected]
  );

  const handleClick = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    onFilesSelected(files);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="relative"
    >
      {/* Hidden input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,video/*"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Upload zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
        className={cn(
          "relative h-56 rounded-2xl border-2 border-dashed transition-all duration-300",
          "flex flex-col items-center justify-center cursor-pointer group",
          "bg-gradient-to-br from-card/60 to-card/20",
          isDragging
            ? "border-violet-500 bg-violet-500/5 scale-[1.01]"
            : "border-white/10 hover:border-violet-500/40 hover:bg-violet-500/[0.02]"
        )}
      >
        {/* Animated gradient background */}
        <div
          className={cn(
            "absolute inset-0 rounded-2xl opacity-0 transition-opacity duration-500",
            isDragging ? "opacity-100" : "group-hover:opacity-60"
          )}
          style={{
            background:
              "radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.08) 0%, transparent 70%)",
          }}
        />

        {/* Content */}
        <div className="relative z-10 text-center px-6">
          {isUploading ? (
            <>
              {uploadStatus === "uploading" && (
                <div className="animate-pulse">
                  <Loader2 className="w-12 h-12 text-violet-400 animate-spin mx-auto" />
                  <p className="mt-4 font-medium text-foreground">
                    Processing files...
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Creating project
                  </p>
                </div>
              )}
              {uploadStatus === "success" && (
                <div className="animate-fade-in">
                  <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto" />
                  <p className="mt-4 font-medium text-emerald-400">
                    Upload Complete!
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Redirecting to Studio...
                  </p>
                </div>
              )}
              {uploadStatus === "error" && (
                <div className="animate-fade-in">
                  <AlertCircle className="w-12 h-12 text-rose-400 mx-auto" />
                  <p className="mt-4 font-medium text-rose-400">
                    Upload Failed
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Please try again
                  </p>
                </div>
              )}
            </>
          ) : (
            <>
              {/* Icon */}
              <div
                className={cn(
                  "w-16 h-16 rounded-2xl mx-auto flex items-center justify-center transition-all duration-300",
                  isDragging
                    ? "bg-violet-500 text-white scale-110"
                    : "bg-white/5 text-muted-foreground group-hover:bg-violet-500/20 group-hover:text-violet-400"
                )}
              >
                <Upload size={28} strokeWidth={1.5} />
              </div>

              {/* Text */}
              <h3 className="mt-5 text-lg font-semibold text-foreground">
                {isDragging ? "Release to upload" : "Drop files here"}
              </h3>
              <p className="text-sm text-muted-foreground mt-1.5">
                or click to browse from your computer
              </p>

              {/* File type badges */}
              <div className="flex gap-2 justify-center mt-5">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 text-xs text-muted-foreground group-hover:bg-violet-500/10 group-hover:text-violet-300 transition-colors">
                  <ImageIcon size={12} />
                  Images
                </span>
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/5 text-xs text-muted-foreground group-hover:bg-emerald-500/10 group-hover:text-emerald-300 transition-colors">
                  <Video size={12} />
                  Videos
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default HeroUpload;
