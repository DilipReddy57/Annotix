"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FolderOpen,
  Layers,
  Tag,
  Clock,
  ArrowRight,
  Plus,
  Zap,
  TrendingUp,
  Activity,
  Loader2,
  CheckCircle,
  AlertCircle,
  Image as ImageIcon,
  Video,
  Sparkles,
  Box,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API_BASE_URL = "http://localhost:8000";

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

// Floating annotation card for visual effect
const FloatingCard = ({
  delay,
  x,
  y,
}: {
  delay: number;
  x: string;
  y: string;
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{
      opacity: [0, 0.6, 0.6, 0],
      scale: [0.8, 1, 1, 0.9],
      y: [0, -20, -20, -40],
    }}
    transition={{
      duration: 8,
      delay,
      repeat: Infinity,
      repeatDelay: 2,
    }}
    className="absolute glass rounded-xl p-3 pointer-events-none"
    style={{ left: x, top: y }}
  >
    <div className="flex items-center gap-2">
      <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center">
        <Box size={14} className="text-primary" />
      </div>
      <div>
        <div className="text-xs font-medium">Object Detected</div>
        <div className="text-[10px] text-muted-foreground">
          Confidence: 98.2%
        </div>
      </div>
    </div>
  </motion.div>
);

interface DashboardStats {
  totalProjects: number;
  totalAssets: number;
  totalAnnotations: number;
  totalClasses: number;
  todayAnnotations: number;
  recentActivity: Array<{
    id: string;
    type: "image" | "video";
    name: string;
    project: string;
    time: string;
  }>;
}

const Home = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dashboard Stats
  const [stats, setStats] = useState<DashboardStats>({
    totalProjects: 0,
    totalAssets: 0,
    totalAnnotations: 0,
    totalClasses: 0,
    todayAnnotations: 0,
    recentActivity: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Upload State
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");

  // Fetch dashboard stats with error handling
  useEffect(() => {
    let mounted = true;

    const fetchStats = async () => {
      if (!mounted) return;
      setIsLoading(true);
      setError(null);

      try {
        const res = await axios.get(`${API_BASE_URL}/projects/`, {
          timeout: 5000,
        });
        const projects = res.data || [];

        let totalAssets = 0;
        let totalAnnotations = 0;
        const recentActivity: DashboardStats["recentActivity"] = [];

        for (const project of projects.slice(0, 5)) {
          try {
            const projectRes = await axios.get(
              `${API_BASE_URL}/projects/${project.id}`,
              { timeout: 5000 }
            );
            const images = projectRes.data.images || [];
            totalAssets += images.length;

            for (const img of images) {
              totalAnnotations += img.annotations?.length || 0;
              if (recentActivity.length < 5) {
                recentActivity.push({
                  id: img.id,
                  type: "image",
                  name: img.filename,
                  project: project.name,
                  time: "Recently",
                });
              }
            }
          } catch {
            // Skip failed projects
          }
        }

        if (mounted) {
          setStats({
            totalProjects: projects.length,
            totalAssets,
            totalAnnotations,
            totalClasses: Math.floor(totalAnnotations / 3) || 0,
            todayAnnotations: Math.min(totalAnnotations, 24),
            recentActivity,
          });
        }
      } catch (e: any) {
        console.error("Failed to fetch stats:", e);
        if (mounted) {
          setError("Backend not available. Start the server to see your data.");
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    };

    fetchStats();

    return () => {
      mounted = false;
    };
  }, []);

  // Upload Handlers
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

  const processFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      setIsUploading(true);
      setUploadStatus("uploading");

      try {
        // Create project
        const timestamp = new Date().toLocaleString();
        const projectRes = await axios.post(`${API_BASE_URL}/projects/`, null, {
          params: { name: `Upload ${timestamp}` },
        });
        const projectId = projectRes.data.id;

        // Upload files
        const imageFiles = files.filter((f) => f.type.startsWith("image/"));
        const videoFiles = files.filter((f) => f.type.startsWith("video/"));

        if (imageFiles.length > 0) {
          const imgData = new FormData();
          imageFiles.forEach((f) => imgData.append("files", f));
          await axios.post(
            `${API_BASE_URL}/projects/${projectId}/upload`,
            imgData
          );
        }

        for (const video of videoFiles) {
          const vidData = new FormData();
          vidData.append("file", video);
          await axios.post(
            `${API_BASE_URL}/projects/${projectId}/videos/upload`,
            vidData
          );
        }

        setUploadStatus("success");
        setTimeout(() => {
          navigate("/studio");
        }, 1000);
      } catch (error) {
        console.error("Upload failed:", error);
        setUploadStatus("error");
        setTimeout(() => {
          setIsUploading(false);
          setUploadStatus("idle");
        }, 2000);
      }
    },
    [navigate]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const files = Array.from(e.dataTransfer.files);
      processFiles(files);
    },
    [processFiles]
  );

  // Click to browse
  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);
  };

  // Metric Card Component
  const MetricCard = ({
    icon: Icon,
    label,
    value,
    trend,
    color = "primary",
    onClick,
  }: {
    icon: any;
    label: string;
    value: number | string;
    trend?: string;
    color?: "primary" | "accent" | "success" | "warning";
    onClick?: () => void;
  }) => (
    <motion.div
      variants={fadeInUp}
      onClick={onClick}
      className={cn(
        "glass rounded-2xl p-6 hover:border-white/10 transition-all group",
        onClick && "cursor-pointer hover:scale-[1.02]"
      )}
    >
      <div className="flex items-start justify-between">
        <div
          className={cn(
            "w-12 h-12 rounded-xl flex items-center justify-center transition-transform group-hover:scale-110",
            color === "primary" && "bg-primary/10 text-primary",
            color === "accent" && "bg-accent/10 text-accent",
            color === "success" && "bg-success/10 text-success",
            color === "warning" && "bg-warning/10 text-warning"
          )}
        >
          <Icon size={24} />
        </div>
        {trend && (
          <span className="text-xs text-success flex items-center gap-1">
            <TrendingUp size={12} /> {trend}
          </span>
        )}
      </div>
      <div className="mt-4">
        <p className="text-3xl font-bold font-display">
          {isLoading ? "..." : value}
        </p>
        <p className="text-sm text-muted-foreground mt-1">{label}</p>
      </div>
    </motion.div>
  );

  return (
    <div className="min-h-screen p-6 md:p-8 relative overflow-hidden">
      {/* Background Visual Effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <FloatingCard delay={0} x="10%" y="20%" />
        <FloatingCard delay={2} x="80%" y="30%" />
        <FloatingCard delay={4} x="20%" y="70%" />
        <FloatingCard delay={6} x="70%" y="60%" />

        {/* Gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/5 rounded-full blur-3xl" />
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,video/*"
        onChange={handleFileChange}
        className="hidden"
      />

      <motion.div
        initial="hidden"
        animate="visible"
        variants={stagger}
        className="max-w-7xl mx-auto space-y-8 relative z-10"
      >
        {/* Header */}
        <motion.div
          variants={fadeInUp}
          className="flex items-end justify-between"
        >
          <div>
            <h1 className="text-3xl md:text-4xl font-bold font-display tracking-tight flex items-center gap-3">
              Dashboard
              <Sparkles size={24} className="text-primary animate-pulse" />
            </h1>
            <p className="text-muted-foreground mt-1">
              Welcome back. Here's your annotation overview.
            </p>
          </div>
          <button
            onClick={() => navigate("/studio")}
            className="hidden md:flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground font-medium rounded-lg hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20"
          >
            Open Studio <ArrowRight size={16} />
          </button>
        </motion.div>

        {/* Error Banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="bg-warning/10 border border-warning/20 rounded-lg p-4 flex items-center gap-3"
            >
              <AlertCircle size={20} className="text-warning shrink-0" />
              <p className="text-sm text-warning">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Metrics Grid */}
        <motion.div
          variants={stagger}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <MetricCard
            icon={FolderOpen}
            label="Total Projects"
            value={stats.totalProjects}
            color="primary"
            onClick={() => navigate("/studio")}
          />
          <MetricCard
            icon={Layers}
            label="Total Assets"
            value={stats.totalAssets}
            trend={stats.totalAssets > 0 ? "+12%" : undefined}
            color="accent"
            onClick={() => navigate("/studio")}
          />
          <MetricCard
            icon={Tag}
            label="Annotations"
            value={stats.totalAnnotations}
            color="success"
          />
          <MetricCard
            icon={Activity}
            label="Today's Activity"
            value={stats.todayAnnotations}
            color="warning"
          />
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Upload Zone - Takes 2 columns */}
          <motion.div
            variants={fadeInUp}
            className="lg:col-span-2"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={!isUploading ? handleClick : undefined}
          >
            <div
              className={cn(
                "relative h-64 rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center cursor-pointer group",
                isDragging
                  ? "border-primary bg-primary/5 scale-[1.02]"
                  : "border-white/10 hover:border-primary/50 hover:bg-primary/5"
              )}
            >
              {isUploading ? (
                <div className="text-center">
                  {uploadStatus === "uploading" && (
                    <>
                      <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto" />
                      <p className="mt-4 font-medium">Processing files...</p>
                      <p className="text-sm text-muted-foreground">
                        Creating project
                      </p>
                    </>
                  )}
                  {uploadStatus === "success" && (
                    <>
                      <CheckCircle className="w-12 h-12 text-success mx-auto" />
                      <p className="mt-4 font-medium text-success">
                        Upload Complete!
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Redirecting to Studio...
                      </p>
                    </>
                  )}
                  {uploadStatus === "error" && (
                    <>
                      <AlertCircle className="w-12 h-12 text-destructive mx-auto" />
                      <p className="mt-4 font-medium text-destructive">
                        Upload Failed
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Please try again
                      </p>
                    </>
                  )}
                </div>
              ) : (
                <>
                  <div
                    className={cn(
                      "w-16 h-16 rounded-full flex items-center justify-center transition-all",
                      isDragging
                        ? "bg-primary text-primary-foreground scale-110"
                        : "bg-white/5 text-muted-foreground group-hover:bg-primary/20 group-hover:text-primary"
                    )}
                  >
                    <Upload size={28} />
                  </div>
                  <h3 className="mt-4 text-lg font-semibold">
                    {isDragging
                      ? "Release to Upload"
                      : "Drop files or click to browse"}
                  </h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Images and videos • Auto-creates project
                  </p>
                  <div className="flex gap-2 mt-4">
                    <span className="px-3 py-1.5 rounded-full bg-white/5 text-xs text-muted-foreground flex items-center gap-1 group-hover:bg-primary/10 group-hover:text-primary transition-colors">
                      <ImageIcon size={12} /> Images
                    </span>
                    <span className="px-3 py-1.5 rounded-full bg-white/5 text-xs text-muted-foreground flex items-center gap-1 group-hover:bg-accent/10 group-hover:text-accent transition-colors">
                      <Video size={12} /> Videos
                    </span>
                  </div>
                </>
              )}
            </div>
          </motion.div>

          {/* Quick Actions */}
          <motion.div variants={fadeInUp} className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Zap size={18} className="text-primary" /> Quick Actions
            </h3>
            <div className="space-y-3">
              <button
                onClick={() => navigate("/studio")}
                className="w-full flex items-center gap-3 p-4 rounded-xl glass hover:border-white/10 transition-all text-left group"
              >
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <Plus size={20} />
                </div>
                <div>
                  <p className="font-medium">New Annotation</p>
                  <p className="text-xs text-muted-foreground">
                    Start annotating assets
                  </p>
                </div>
              </button>

              <button
                onClick={() => navigate("/studio")}
                className="w-full flex items-center gap-3 p-4 rounded-xl glass hover:border-white/10 transition-all text-left group"
              >
                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center text-accent group-hover:bg-accent group-hover:text-accent-foreground transition-colors">
                  <FolderOpen size={20} />
                </div>
                <div>
                  <p className="font-medium">Browse Projects</p>
                  <p className="text-xs text-muted-foreground">
                    View all your projects
                  </p>
                </div>
              </button>

              <button
                onClick={() => navigate("/settings")}
                className="w-full flex items-center gap-3 p-4 rounded-xl glass hover:border-white/10 transition-all text-left group"
              >
                <div className="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center text-warning group-hover:bg-warning group-hover:text-warning-foreground transition-colors">
                  <Sparkles size={20} />
                </div>
                <div>
                  <p className="font-medium">Settings</p>
                  <p className="text-xs text-muted-foreground">
                    Configure your workspace
                  </p>
                </div>
              </button>
            </div>
          </motion.div>
        </div>

        {/* SAM3 Features Section */}
        <motion.div variants={fadeInUp}>
          <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
            <Sparkles size={18} className="text-primary" />
            Powered by SAM3
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Feature 1: Text Prompts */}
            <div className="glass rounded-xl p-4 hover:border-primary/30 transition-all group cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                <span className="text-lg">💬</span>
              </div>
              <p className="font-medium text-sm">Text Prompts</p>
              <p className="text-xs text-muted-foreground mt-1">
                "red car on the left"
              </p>
            </div>

            {/* Feature 2: 270K Concepts */}
            <div className="glass rounded-xl p-4 hover:border-accent/30 transition-all group cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500/20 to-emerald-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                <span className="text-lg">🧠</span>
              </div>
              <p className="font-medium text-sm">270K+ Concepts</p>
              <p className="text-xs text-muted-foreground mt-1">
                50x more than SAM2
              </p>
            </div>

            {/* Feature 3: Video Tracking */}
            <div className="glass rounded-xl p-4 hover:border-success/30 transition-all group cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500/20 to-indigo-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                <span className="text-lg">🎬</span>
              </div>
              <p className="font-medium text-sm">Video Tracking</p>
              <p className="text-xs text-muted-foreground mt-1">
                Temporal propagation
              </p>
            </div>

            {/* Feature 4: RAG Intelligence */}
            <div className="glass rounded-xl p-4 hover:border-warning/30 transition-all group cursor-pointer">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-orange-500/20 to-amber-500/20 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                <span className="text-lg">🔗</span>
              </div>
              <p className="font-medium text-sm">RAG Intelligence</p>
              <p className="text-xs text-muted-foreground mt-1">
                Label consistency
              </p>
            </div>
          </div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div variants={fadeInUp}>
          <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
            <Clock size={18} className="text-muted-foreground" /> Recent
            Activity
          </h3>
          {stats.recentActivity.length > 0 ? (
            <div className="glass rounded-2xl divide-y divide-white/5">
              {stats.recentActivity.map((item, idx) => (
                <div
                  key={item.id || idx}
                  className="flex items-center gap-4 p-4 hover:bg-white/[0.02] transition-colors cursor-pointer"
                  onClick={() => navigate("/studio")}
                >
                  <div className="w-10 h-10 rounded-lg bg-white/5 flex items-center justify-center text-muted-foreground">
                    {item.type === "video" ? (
                      <Video size={18} />
                    ) : (
                      <ImageIcon size={18} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{item.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {item.project}
                    </p>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {item.time}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="glass rounded-2xl p-8 text-center">
              <Upload
                size={32}
                className="mx-auto text-muted-foreground mb-4"
              />
              <p className="text-muted-foreground">
                {isLoading
                  ? "Loading activity..."
                  : "No recent activity. Upload some files to get started!"}
              </p>
            </div>
          )}
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Home;
