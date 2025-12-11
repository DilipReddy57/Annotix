import { useState, useEffect, useRef, useCallback } from "react";
import {
  Search,
  Filter,
  Plus,
  Video,
  Wand2,
  Upload,
  ArrowLeft,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";
import axios from "axios";
import { motion } from "framer-motion";

import { api, API_BASE_URL } from "../../api/client";
import AnnotationEditor from "../Editor/AnnotationEditor";
import VideoPlayer from "../Video/VideoPlayer";
import { cn } from "@/lib/utils";

const API_URL = `${API_BASE_URL}/api/projects`;

// Animation Variants
const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const staggerGrid = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.05 },
  },
};

const ProjectView = () => {
  const [projects, setProjects] = useState<any[]>([]);
  const [selectedProject, setSelectedProject] = useState<any>(null);
  const [images, setImages] = useState<any[]>([]);
  const [videos, setVideos] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Editor State
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [selectedVideo, setSelectedVideo] = useState<any>(null);

  // Video Annotation State
  const [videoPrompt, setVideoPrompt] = useState("");
  const [videoAnnotations, setVideoAnnotations] = useState<any[]>([]);
  const [videoProcessing, setVideoProcessing] = useState(false);

  // Upload State
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // New Project State
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDesc, setNewProjectDesc] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  // Upload handlers
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

  const processUpload = useCallback(
    async (files: File[]) => {
      if (!selectedProject || files.length === 0) return;

      setIsUploading(true);
      setUploadStatus("uploading");

      try {
        const imageFiles = files.filter((f) => f.type.startsWith("image/"));
        const videoFiles = files.filter((f) => f.type.startsWith("video/"));

        if (imageFiles.length > 0) {
          const formData = new FormData();
          imageFiles.forEach((f) => formData.append("files", f));
          await axios.post(`${API_URL}/${selectedProject.id}/upload`, formData);
        }

        for (const video of videoFiles) {
          const formData = new FormData();
          formData.append("file", video);
          await axios.post(
            `${API_URL}/${selectedProject.id}/videos/upload`,
            formData
          );
        }

        setUploadStatus("success");
        // Refresh project data
        await selectProject(selectedProject);
        setTimeout(() => setUploadStatus("idle"), 2000);
      } catch (error) {
        console.error("Upload failed:", error);
        setUploadStatus("error");
        setTimeout(() => setUploadStatus("idle"), 3000);
      } finally {
        setIsUploading(false);
      }
    },
    [selectedProject]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files);
      processUpload(files);
    },
    [processUpload]
  );

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processUpload(files);
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    setIsLoading(true);
    try {
      const res = await axios.get(API_URL);
      setProjects(res.data || []);
      if (res.data.length > 0) {
        await selectProject(res.data[0]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  const createProject = async () => {
    if (!newProjectName.trim()) return;
    setIsCreating(true);
    try {
      const res = await axios.post(
        `${API_URL}/?name=${encodeURIComponent(
          newProjectName
        )}&description=${encodeURIComponent(newProjectDesc)}`
      );
      const newProject = res.data;
      setProjects((prev) => [...prev, newProject]);
      await selectProject(newProject);
      setShowNewProject(false);
      setNewProjectName("");
      setNewProjectDesc("");
    } catch (e) {
      console.error("Failed to create project:", e);
    } finally {
      setIsCreating(false);
    }
  };

  const selectProject = async (p: any) => {
    setSelectedProject(p);
    setSelectedVideo(null);
    setVideoAnnotations([]);
    try {
      const res = await axios.get(`${API_URL}/${p.id}/images`);
      setImages(res.data.images || []);
    } catch (e) {
      setImages([]);
    }

    try {
      const vidRes = await axios.get(`${API_URL}/${p.id}/videos`);
      setVideos(vidRes.data);
    } catch (e) {
      setVideos([]);
    }
  };

  const openEditor = (img: any) => {
    const fullUrl = `http://localhost:8000/data/uploads/${selectedProject.id}/${img.filename}`;
    setSelectedImage({ ...img, url: fullUrl });
  };

  const openVideoPlayer = (vid: any) => {
    setSelectedVideo(vid);
    setVideoAnnotations([]);
  };

  const handleVideoAnnotate = async () => {
    if (!selectedVideo || !videoPrompt.trim()) return;
    setVideoProcessing(true);
    try {
      const result = await api.annotateVideo(
        selectedProject.id,
        selectedVideo.id,
        videoPrompt
      );
      setVideoAnnotations(result.annotations || []);
    } catch (e) {
      console.error(e);
    } finally {
      setVideoProcessing(false);
    }
  };

  const allAssets = [
    ...images.map((i) => ({ ...i, assetType: "image" })),
    ...videos.map((v) => ({ ...v, assetType: "video" })),
  ];

  // ========== FULL SCREEN EDITOR ==========
  if (selectedImage) {
    return (
      <div className="absolute inset-0 z-10 bg-background/95 backdrop-blur-xl flex flex-col">
        <div className="h-14 border-b border-white/5 bg-card/20 backdrop-blur-md flex items-center px-6 justify-between shrink-0">
          <button
            onClick={() => setSelectedImage(null)}
            className="text-white hover:text-primary transition-colors text-sm font-medium flex items-center gap-2"
          >
            <ArrowLeft size={16} /> Back to Studio
          </button>
          <span className="text-sm font-mono text-muted-foreground">
            {selectedImage.filename}
          </span>
          <div className="w-20" />
        </div>
        <div className="flex-1 relative">
          <AnnotationEditor
            imageUrl={selectedImage.url || ""}
            initialAnnotations={selectedImage.annotations}
            onSave={() => console.log("Saved")}
            projectId={selectedProject?.id || ""}
            imageId={selectedImage.id}
          />
        </div>
      </div>
    );
  }

  // ========== FULL SCREEN VIDEO ==========
  if (selectedVideo) {
    const videoUrl = `http://localhost:8000/data/uploads/${selectedProject.id}/${selectedVideo.filename}`;
    return (
      <div className="absolute inset-0 z-10 bg-background/95 backdrop-blur-xl flex flex-col">
        <div className="h-14 border-b border-white/5 bg-card/20 backdrop-blur-md flex items-center px-6 justify-between shrink-0">
          <button
            onClick={() => setSelectedVideo(null)}
            className="text-white hover:text-primary transition-colors text-sm font-medium flex items-center gap-2"
          >
            <ArrowLeft size={16} /> Back to Studio
          </button>
          <span className="text-sm font-mono text-muted-foreground">
            {selectedVideo.filename}
          </span>
          <div className="w-20" />
        </div>
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 bg-black/80 flex items-center justify-center p-4">
            <VideoPlayer src={videoUrl} annotations={videoAnnotations} />
          </div>
          <div className="w-80 border-l border-white/5 bg-card/20 backdrop-blur-md p-6 space-y-6 overflow-y-auto">
            <div>
              <h3 className="text-white font-bold mb-2 flex items-center gap-2">
                <Wand2 size={16} className="text-primary" /> Smart Tracking
              </h3>
              <p className="text-xs text-muted-foreground mb-4">
                Enter an object name to auto-detect and track across frames.
              </p>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={videoPrompt}
                  onChange={(e) => setVideoPrompt(e.target.value)}
                  placeholder="e.g., 'car', 'person'..."
                  className="flex-1 bg-background/50 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary/50 transition-colors"
                />
                <button
                  onClick={handleVideoAnnotate}
                  disabled={videoProcessing}
                  className={cn(
                    "px-4 py-2 bg-primary text-background font-bold rounded-lg transition-all flex items-center gap-2",
                    videoProcessing
                      ? "opacity-50 cursor-wait"
                      : "hover:bg-primary/90"
                  )}
                >
                  {videoProcessing ? (
                    <span className="animate-spin">⏳</span>
                  ) : (
                    <Wand2 size={16} />
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ========== MAIN GALLERY VIEW ==========
  return (
    <div className="h-full flex">
      {/* Sidebar - Project List */}
      <div className="w-64 shrink-0 border-r border-white/10 bg-card/30 backdrop-blur-md flex flex-col">
        <div className="p-4 border-b border-white/10">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Upload size={16} className="text-primary" />
            Projects
          </h3>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {projects.map((project) => (
            <button
              key={project.id}
              onClick={() => selectProject(project)}
              className={cn(
                "w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm",
                selectedProject?.id === project.id
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted-foreground hover:bg-white/5 hover:text-white"
              )}
            >
              <div className="font-medium truncate">{project.name}</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {project.description || "No description"}
              </div>
            </button>
          ))}
          {projects.length === 0 && !isLoading && (
            <div className="text-center py-8 text-muted-foreground text-xs">
              No projects yet
            </div>
          )}
        </div>
        <div className="p-3 border-t border-white/10">
          {showNewProject ? (
            <div className="space-y-2">
              <input
                type="text"
                placeholder="Project name"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                className="w-full px-3 py-2 bg-card/50 border border-white/10 rounded-lg text-sm text-white placeholder:text-muted-foreground focus:outline-none focus:border-primary/50"
                autoFocus
              />
              <input
                type="text"
                placeholder="Description (optional)"
                value={newProjectDesc}
                onChange={(e) => setNewProjectDesc(e.target.value)}
                className="w-full px-3 py-2 bg-card/50 border border-white/10 rounded-lg text-sm text-white placeholder:text-muted-foreground focus:outline-none focus:border-primary/50"
              />
              <div className="flex gap-2">
                <button
                  onClick={() => setShowNewProject(false)}
                  className="flex-1 px-3 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg text-sm transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={createProject}
                  disabled={isCreating || !newProjectName.trim()}
                  className="flex-1 px-3 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isCreating ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Plus size={14} />
                  )}
                  Create
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={() => setShowNewProject(true)}
              className="w-full px-3 py-2 bg-primary/10 hover:bg-primary/20 text-primary rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Plus size={14} />
              New Project
            </button>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div
        className={cn(
          "flex-1 flex flex-col p-6 space-y-6 transition-colors overflow-auto",
          isDragging && "bg-primary/5 ring-2 ring-primary/30 ring-inset"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Hidden file input */}
        <input
          id="studio-file-upload"
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,video/*"
          onChange={handleFileSelect}
          className="hidden"
        />
        {/* Studio Header */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 shrink-0"
        >
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-3">
              {selectedProject?.name || "Loading..."}
              <span className="px-2.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-[10px] text-primary font-bold uppercase tracking-wider">
                Active
              </span>
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {isLoading
                ? "Loading assets..."
                : `${allAssets.length} assets ready for annotation.`}
            </p>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* Upload Status Indicator */}
            {uploadStatus !== "idle" && (
              <div
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium",
                  uploadStatus === "uploading" && "bg-primary/10 text-primary",
                  uploadStatus === "success" && "bg-success/10 text-success",
                  uploadStatus === "error" &&
                    "bg-destructive/10 text-destructive"
                )}
              >
                {uploadStatus === "uploading" && (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Uploading...
                  </>
                )}
                {uploadStatus === "success" && (
                  <>
                    <CheckCircle size={14} /> Uploaded!
                  </>
                )}
                {uploadStatus === "error" && (
                  <>
                    <XCircle size={14} /> Failed
                  </>
                )}
              </div>
            )}
            <div className="relative group">
              <Search
                className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground group-focus-within:text-primary transition-colors"
                size={14}
              />
              <input
                type="text"
                placeholder="Filter assets..."
                className="w-48 bg-card/30 backdrop-blur-sm border border-white/10 rounded-lg py-2 pl-9 pr-4 text-xs focus:outline-none focus:border-primary/50 text-white transition-colors"
              />
            </div>
            <button className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-sm text-white transition-colors flex items-center gap-2 backdrop-blur-md">
              <Filter size={16} /> Filters
            </button>
            <label
              htmlFor="studio-file-upload"
              className={cn(
                "px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground font-bold rounded-lg shadow-lg shadow-primary/20 transition-all flex items-center gap-2 cursor-pointer",
                isUploading &&
                  "opacity-50 cursor-not-allowed pointer-events-none"
              )}
            >
              {isUploading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Plus size={16} />
              )}
              {isUploading ? "Uploading..." : "Upload"}
            </label>
          </div>
        </motion.div>

        {/* Gallery Grid */}
        <div className="flex-1 overflow-y-auto pr-2 pb-10">
          {allAssets.length === 0 && !isLoading ? (
            // Empty State
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="h-full flex flex-col items-center justify-center text-center p-12"
            >
              <div className="w-24 h-24 rounded-full bg-card/30 flex items-center justify-center mb-6">
                <Upload size={40} className="text-muted-foreground" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2">
                No Assets Yet
              </h3>
              <p className="text-muted-foreground max-w-md mb-6">
                Drop images or videos on the Home page, or click "Upload" to add
                media to this project.
              </p>
              <button className="px-6 py-3 bg-primary hover:bg-primary/90 text-background font-bold rounded-lg transition-all flex items-center gap-2">
                <Plus size={18} /> Add Your First Asset
              </button>
            </motion.div>
          ) : (
            <motion.div
              initial="hidden"
              animate="visible"
              variants={staggerGrid}
              className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-5"
            >
              {allAssets.map((asset) => (
                <motion.div
                  key={asset.id}
                  variants={fadeIn}
                  onClick={() =>
                    asset.assetType === "video"
                      ? openVideoPlayer(asset)
                      : openEditor(asset)
                  }
                  className="group aspect-[4/3] bg-card/20 backdrop-blur-sm rounded-xl border border-white/5 overflow-hidden cursor-pointer relative hover:ring-2 hover:ring-primary/50 transition-all hover:bg-card/40 hover:scale-[1.02]"
                >
                  {/* Media Thumbnail */}
                  <div className="absolute inset-0">
                    {asset.assetType === "video" ? (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-purple-500/10 to-transparent text-muted-foreground group-hover:text-white transition-colors">
                        <Video size={32} />
                        <span className="text-[10px] mt-2 uppercase tracking-wider">
                          Video
                        </span>
                      </div>
                    ) : (
                      <img
                        src={`http://localhost:8000/data/uploads/${selectedProject?.id}/${asset.filename}`}
                        alt={asset.filename}
                        className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity duration-300"
                        loading="lazy"
                      />
                    )}
                  </div>

                  {/* Hover Overlay */}
                  <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 via-black/50 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-200">
                    <p className="text-xs font-medium text-white truncate">
                      {asset.filename}
                    </p>
                    <p className="text-[10px] text-muted-foreground">
                      {asset.annotations?.length || 0} annotations
                    </p>
                  </div>

                  {/* Top Badges */}
                  <div className="absolute top-2 right-2 flex gap-1.5">
                    {asset.assetType === "video" && (
                      <div className="p-1.5 rounded-md bg-black/60 text-white backdrop-blur-md">
                        <Video size={10} />
                      </div>
                    )}
                    {/* Status Badges */}
                    {asset.status === "pending" && (
                      <div className="px-2 py-1 rounded-md bg-amber-500/20 text-amber-400 backdrop-blur-md text-[9px] font-medium">
                        Pending
                      </div>
                    )}
                    {asset.status === "processing" && (
                      <div className="px-2 py-1 rounded-md bg-blue-500/20 text-blue-400 backdrop-blur-md text-[9px] font-medium flex items-center gap-1">
                        <Loader2 size={8} className="animate-spin" />
                        Processing
                      </div>
                    )}
                    {asset.status === "completed" && (
                      <div className="px-2 py-1 rounded-md bg-emerald-500/20 text-emerald-400 backdrop-blur-md text-[9px] font-medium flex items-center gap-1">
                        <CheckCircle size={8} />
                        Done
                      </div>
                    )}
                    {asset.status === "error" && (
                      <div className="px-2 py-1 rounded-md bg-red-500/20 text-red-400 backdrop-blur-md text-[9px] font-medium flex items-center gap-1">
                        <XCircle size={8} />
                        Error
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}

              {/* Add New Card - Upload Zone */}
              <motion.label
                htmlFor="studio-file-upload"
                variants={fadeIn}
                className={cn(
                  "aspect-[4/3] rounded-xl border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all backdrop-blur-sm",
                  isDragging
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-white/10 text-muted-foreground hover:text-primary hover:border-primary/40 hover:bg-primary/5"
                )}
              >
                <Upload size={28} />
                <span className="text-xs font-medium mt-2">
                  {isDragging ? "Drop files here" : "Add New"}
                </span>
                <span className="text-[10px] text-muted-foreground mt-1">
                  Click or drag files
                </span>
              </motion.label>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectView;
