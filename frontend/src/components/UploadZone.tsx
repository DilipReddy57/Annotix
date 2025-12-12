import React, { useState, useRef, DragEvent, ChangeEvent } from "react";
import { Upload, FileVideo, Image as ImageIcon, Loader2 } from "lucide-react";
import axios from "axios";
import { API_BASE_URL } from "../api/client";

interface UploadZoneProps {
  onUploadComplete?: () => void;
}

const UploadZone: React.FC<UploadZoneProps> = ({ onUploadComplete }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = async (files: FileList) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", files[0]);

    try {
      // 1. Upload
      await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // 2. Poll for status (simplified for demo)
      setTimeout(() => {
        setIsUploading(false);
        if (onUploadComplete) onUploadComplete();
      }, 2000);
    } catch (error) {
      console.error("Upload failed", error);
      setIsUploading(false);
    }
  };

  return (
    <div className="glass rounded-2xl p-8 border-dashed border-2 border-white/10 hover:border-primary/50 transition-colors">
      <div
        className={`flex flex-col items-center justify-center h-64 cursor-pointer transition-all ${
          isDragging ? "scale-[1.02] bg-white/5" : ""
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            if (e.target.files) handleFiles(e.target.files);
          }}
        />

        {isUploading ? (
          <div className="flex flex-col items-center gap-4">
            <Loader2 size={48} className="text-primary animate-spin" />
            <p className="text-lg font-medium">Processing Media...</p>
            <span className="text-sm text-gray-400">
              Running SAM 3 Segmentation
            </span>
          </div>
        ) : (
          <>
            <div className="w-20 h-20 bg-primary/10 rounded-full flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
              <Upload size={32} className="text-primary" />
            </div>
            <h3 className="text-2xl font-bold mb-2">Upload Data</h3>
            <p className="text-gray-400 mb-6">
              Drag & drop images or videos here
            </p>
            <div className="flex gap-4 text-sm text-gray-500">
              <span className="flex items-center gap-2">
                <ImageIcon size={16} /> JPG, PNG
              </span>
              <span className="flex items-center gap-2">
                <FileVideo size={16} /> MP4, AVI
              </span>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default UploadZone;
