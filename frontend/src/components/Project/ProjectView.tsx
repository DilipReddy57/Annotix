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
  X,
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
  const [isDeleting, setIsDeleting] = useState<string | null>(null);

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

  // Pagination State
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [totalImages, setTotalImages] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  // Folder Navigation State
  const [viewMode, setViewMode] = useState<"grid" | "folder">("grid");
  const [currentPath, setCurrentPath] = useState<string[]>([]);
  const [isSyncing, setIsSyncing] = useState(false);

  // New Project State
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDesc, setNewProjectDesc] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  // Delete Project handler
  const deleteProject = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Don't select the project when clicking delete

    if (
      !confirm(
        "Are you sure you want to delete this project? This will delete all images and annotations."
      )
    ) {
      return;
    }

    setIsDeleting(projectId);
    try {
      await axios.delete(`${API_URL}/${projectId}`);
      setProjects((prev) => prev.filter((p) => p.id !== projectId));

      // If we deleted the selected project, select another one
      if (selectedProject?.id === projectId) {
        const remaining = projects.filter((p) => p.id !== projectId);
        if (remaining.length > 0) {
          await selectProject(remaining[0]);
        } else {
          setSelectedProject(null);
          setImages([]);
          setVideos([]);
        }
      }
    } catch (error) {
      console.error("Failed to delete project:", error);
      alert("Failed to delete project");
    } finally {
      setIsDeleting(null);
    }
  };

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
      const res = await axios.get(`${API_URL}?t=${Date.now()}`);
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

  const openEditor = (img: any) => {
    const fullUrl = `${API_BASE_URL}/api/projects/${selectedProject.id}/image/${
      img.filename
    }?t=${Date.now()}`;
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
    const videoUrl = `${API_BASE_URL}/api/projects/${
      selectedProject.id
    }/image/${selectedVideo.filename}?t=${Date.now()}`;
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

  // ========== PAGINATION LOGIC ==========

  const loadMoreImages = async () => {
    if (!selectedProject || !hasMore || isLoadingMore) return;
    setIsLoadingMore(true);
    try {
      const nextPage = page + 1;
      const res = await axios.get(
        `${API_URL}/${selectedProject.id}/images?page=${nextPage}&limit=50`
      );

      if (res.data.images && res.data.images.length > 0) {
        setImages((prev) => [...prev, ...res.data.images]);
        setPage(nextPage);
        setHasMore(res.data.images.length === 50); // If less than limit, no more
      } else {
        setHasMore(false);
      }
    } catch (e) {
      console.error("Failed to load more images", e);
    } finally {
      setIsLoadingMore(false);
    }
  };

  const selectProject = async (p: any) => {
    setSelectedProject(p);
    setSelectedVideo(null);
    setVideoAnnotations([]);
    setPage(1);
    setHasMore(true);
    setImages([]); // Clear immediate

    try {
      // Initial fetch (Page 1)
      const res = await axios.get(`${API_URL}/${p.id}/images?page=1&limit=50`);
      setImages(res.data.images || []);
      setTotalImages(res.data.total || 0);
      setHasMore((res.data.images?.length || 0) === 50);
    } catch (e) {
      console.error("Failed to load project images", e);
      setImages([]);
    }

    try {
      const vidRes = await axios.get(`${API_URL}/${p.id}/videos`);
      setVideos(vidRes.data);
    } catch (e) {
      setVideos([]);
    }
  };

  // ========== FOLDER NAVIGATION LOGIC ==========

  // Group assets by folder for the current path
  const getFolderContent = () => {
    if (viewMode === "grid") return { folders: [], files: allAssets };

    const currentPathStr =
      currentPath.length > 0 ? currentPath.join("/") + "/" : "";
    const folders = new Set<string>();
    const files: any[] = [];

    allAssets.forEach((asset) => {
      // Normalize path separators
      const cleanFilename = asset.filename.replace(/\\/g, "/");

      if (!cleanFilename.startsWith(currentPathStr)) return;

      const relativePath = cleanFilename.slice(currentPathStr.length);
      const parts = relativePath.split("/");

      if (parts.length > 1) {
        // It's in a subfolder
        folders.add(parts[0]);
      } else {
        // It's a file in the current directory
        files.push(asset);
      }
    });

    return {
      folders: Array.from(folders).sort(),
      files: files,
    };
  };

  const { folders, files: currentFiles } = getFolderContent();

  const handleSyncProject = async () => {
    if (!selectedProject || isSyncing) return;
    setIsSyncing(true);
    try {
      const res = await axios.post(`${API_URL}/${selectedProject.id}/sync`);
      // Reload project to get new files
      await selectProject(selectedProject);
      alert(
        `Sync complete: ${res.data.added} new files added, ${res.data.updated} repaired.`
      );
    } catch (e) {
      console.error("Sync failed", e);
      alert("Sync failed. Check console.");
    } finally {
      setIsSyncing(false);
    }
  };

  // ========== MAIN GALLERY VIEW ==========
  return (
    <div className="h-full flex">
      {/* Sidebar - Project List */}
      <div className="w-64 shrink-0 border-r border-white/10 bg-card/30 backdrop-blur-md flex flex-col">
        {/* ... sidebar content same as before ... */}
        <div className="p-4 border-b border-white/10">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Upload size={16} className="text-primary" />
            Projects
          </h3>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {projects.map((project) => (
            <div
              key={project.id}
              className={cn(
                "group relative w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm cursor-pointer",
                selectedProject?.id === project.id
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted-foreground hover:bg-white/5 hover:text-white"
              )}
              onClick={() => selectProject(project)}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{project.name}</div>
                  <div className="text-xs text-muted-foreground mt-0.5 truncate">
                    {project.description || "No description"}
                  </div>
                </div>
                {/* Delete button */}
                <button
                  onClick={(e) => deleteProject(project.id, e)}
                  disabled={isDeleting === project.id}
                  className={cn(
                    "shrink-0 p-1 rounded-md opacity-0 group-hover:opacity-100 transition-all",
                    "hover:bg-red-500/20 hover:text-red-400",
                    isDeleting === project.id && "opacity-100"
                  )}
                  title="Delete project"
                >
                  {isDeleting === project.id ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <X size={14} />
                  )}
                </button>
              </div>
            </div>
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
              {/* ... new project form ... */}
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
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-3">
                {selectedProject?.name || "Loading..."}
                <span className="px-2.5 py-1 rounded-full bg-primary/10 border border-primary/20 text-[10px] text-primary font-bold uppercase tracking-wider">
                  Active
                </span>
              </h2>
              {/* Sync Button */}
              <button
                onClick={handleSyncProject}
                disabled={isSyncing}
                className="p-1.5 rounded-md hover:bg-white/10 text-muted-foreground hover:text-white transition-colors"
                title="Rescan project files (Fix missing assets)"
              >
                <div
                  className={cn(
                    "transition-transform",
                    isSyncing && "animate-spin"
                  )}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                    <path d="M3 3v5h5" />
                    <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
                    <path d="M16 21h5v-5" />
                  </svg>
                </div>
              </button>
            </div>

            <p className="text-sm text-muted-foreground mt-1">
              {isLoading
                ? "Loading assets..."
                : `${
                    totalImages > 0 ? totalImages : allAssets.length
                  } assets available (${allAssets.length} loaded)`}
            </p>
          </div>

          <div className="flex items-center gap-3 flex-wrap">
            {/* View Mode Toggle */}
            <div className="flex bg-white/5 rounded-lg p-1 border border-white/10">
              <button
                onClick={() => setViewMode("grid")}
                className={cn(
                  "p-1.5 rounded-md transition-all",
                  viewMode === "grid"
                    ? "bg-primary text-white shadow-sm"
                    : "text-muted-foreground hover:text-white"
                )}
                title="Grid View"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect width="7" height="7" x="3" y="3" rx="1" />
                  <rect width="7" height="7" x="14" y="3" rx="1" />
                  <rect width="7" height="7" x="14" y="14" rx="1" />
                  <rect width="7" height="7" x="3" y="14" rx="1" />
                </svg>
              </button>
              <button
                onClick={() => setViewMode("folder")}
                className={cn(
                  "p-1.5 rounded-md transition-all",
                  viewMode === "folder"
                    ? "bg-primary text-white shadow-sm"
                    : "text-muted-foreground hover:text-white"
                )}
                title="Folder View"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
                </svg>
              </button>
            </div>

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

        {/* Gallery Content */}
        <div className="flex-1 overflow-y-auto pr-2 pb-10">
          {/* Breadcrumbs for Folder View */}
          {viewMode === "folder" && (
            <div className="flex items-center gap-2 mb-4 text-sm text-muted-foreground">
              <button
                onClick={() => setCurrentPath([])}
                className="hover:text-primary transition-colors flex items-center gap-1"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                  <polyline points="9 22 9 12 15 12 15 22" />
                </svg>
                Root
              </button>
              {currentPath.map((folder, index) => (
                <div key={folder} className="flex items-center gap-2">
                  <span>/</span>
                  <button
                    onClick={() =>
                      setCurrentPath(currentPath.slice(0, index + 1))
                    }
                    className="hover:text-primary transition-colors font-medium text-white"
                  >
                    {folder}
                  </button>
                </div>
              ))}
            </div>
          )}

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
            </motion.div>
          ) : (
            <>
              <motion.div
                initial="hidden"
                animate="visible"
                variants={staggerGrid}
                className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-5"
              >
                {/* Folder Cards (Only in Folder View) */}
                {viewMode === "folder" &&
                  folders.map((folder) => (
                    <motion.div
                      key={folder}
                      variants={fadeIn}
                      onClick={() => setCurrentPath([...currentPath, folder])}
                      className="aspect-[4/3] bg-card/10 backdrop-blur-sm rounded-xl border border-dashed border-white/20 flex flex-col items-center justify-center cursor-pointer hover:bg-card/30 hover:border-primary/50 transition-all group"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="40"
                        height="40"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="text-primary/70 group-hover:text-primary group-hover:scale-110 transition-all mb-3"
                      >
                        <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />
                      </svg>
                      <span className="text-sm font-medium text-white group-hover:text-primary transition-colors">
                        {folder}
                      </span>
                    </motion.div>
                  ))}

                {/* File Cards */}
                {(viewMode === "grid" ? allAssets : currentFiles).map(
                  (asset) => (
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
                            src={`${API_BASE_URL}/api/projects/${
                              selectedProject?.id
                            }/image/${asset.filename}?t=${Date.now()}`}
                            alt={asset.filename}
                            className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity duration-300"
                            loading="lazy"
                          />
                        )}
                      </div>

                      {/* Hover Overlay */}
                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 via-black/50 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-200">
                        <p
                          className="text-xs font-medium text-white truncate"
                          title={asset.filename}
                        >
                          {asset.filename.split("/").pop()}
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
                  )
                )}
              </motion.div>

              {/* Load More Button */}
              {hasMore && viewMode === "grid" && (
                <div className="mt-8 flex justify-center pb-8">
                  <button
                    onClick={loadMoreImages}
                    disabled={isLoadingMore}
                    className="px-6 py-2.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full text-sm font-medium text-white transition-all flex items-center gap-2 group"
                  >
                    {isLoadingMore ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className="group-hover:translate-y-0.5 transition-transform"
                      >
                        <path d="m6 9 6 6 6-6" />
                      </svg>
                    )}
                    {isLoadingMore ? "Loading..." : "Load More"}
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ProjectView;
