"use client";

import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  FolderOpen,
  Layers,
  Tag,
  Activity,
  Sparkles,
  ArrowRight,
  AlertCircle,
} from "lucide-react";
import { API_BASE_URL } from "../api/client";
import AnnotationRain from "./ui/annotation-rain";
import StatCard from "./dashboard/StatCard";
import HeroUpload from "./dashboard/HeroUpload";
import SystemStatus from "./dashboard/SystemStatus";
import QuickActions from "./dashboard/QuickActions";
import RecentActivity from "./dashboard/RecentActivity";
import DatasetImport from "./dashboard/DatasetImport";

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

  // State
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
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");

  // Fetch dashboard stats
  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const fetchStats = async () => {
      if (stats.totalAssets === 0 && stats.totalProjects === 0) {
        setIsLoading(true);
      }
      setError(null);

      try {
        const res = await axios.get(`${API_BASE_URL}/api/projects/stats`, {
          signal: controller.signal,
          timeout: 10000,
        });

        if (isMounted && res.data) {
          setStats({
            totalProjects: res.data.totalProjects || 0,
            totalAssets: res.data.totalAssets || 0,
            totalAnnotations: res.data.totalAnnotations || 0,
            totalClasses: res.data.totalClasses || 0,
            todayAnnotations: res.data.todayAnnotations || 0,
            recentActivity: res.data.recentActivity || [],
          });
        }
      } catch (e: any) {
        if (axios.isCancel(e)) return;
        if (
          isMounted &&
          (e.code === "ECONNREFUSED" || e.code === "ERR_NETWORK")
        ) {
          setError("Backend not available. Start the server to see your data.");
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchStats();
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, []);

  // File upload handler
  const handleFilesSelected = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      setIsUploading(true);
      setUploadStatus("uploading");

      try {
        const timestamp = new Date().toLocaleString();
        const projectRes = await axios.post(
          `${API_BASE_URL}/api/projects/`,
          null,
          { params: { name: `Upload ${timestamp}` } }
        );
        const projectId = projectRes.data.id;

        const imageFiles = files.filter((f) => f.type.startsWith("image/"));
        const videoFiles = files.filter((f) => f.type.startsWith("video/"));

        if (imageFiles.length > 0) {
          const imgData = new FormData();
          imageFiles.forEach((f) => imgData.append("files", f));
          await axios.post(
            `${API_BASE_URL}/api/projects/${projectId}/upload`,
            imgData
          );
        }

        for (const video of videoFiles) {
          const vidData = new FormData();
          vidData.append("file", video);
          await axios.post(
            `${API_BASE_URL}/api/projects/${projectId}/videos/upload`,
            vidData
          );
        }

        setUploadStatus("success");
        setTimeout(() => navigate("/studio"), 1000);
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

  // Dataset URL import handler
  const handleDatasetImport = useCallback(
    async (url: string, source: string) => {
      setIsUploading(true);
      setUploadStatus("uploading");

      try {
        // Call backend to import dataset from URL
        const response = await axios.post(
          `${API_BASE_URL}/api/projects/import-dataset`,
          {
            url,
            source,
          }
        );

        const { project_id, project_name, message } = response.data;

        setUploadStatus("success");

        // Show success message
        alert(
          `✅ Import Started!\n\n${message}\n\nProject ID: ${project_id}\n\nNote: Large datasets download in the background. Check the project in Studio after a few moments.`
        );

        setTimeout(() => {
          setIsUploading(false);
          setUploadStatus("idle");
          // Reload to refresh stats
          window.location.reload();
        }, 1000);
      } catch (error: any) {
        console.error("Dataset import failed:", error);
        const errorMsg =
          error.response?.data?.detail || error.message || "Unknown error";
        alert(`❌ Import Failed!\n\n${errorMsg}`);
        setUploadStatus("error");
        setTimeout(() => {
          setIsUploading(false);
          setUploadStatus("idle");
        }, 2000);
        throw new Error(
          error.response?.data?.detail || "Failed to import dataset"
        );
      }
    },
    [navigate]
  );

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Annotation Rain Background */}
      <AnnotationRain />

      {/* Content overlay gradient */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "linear-gradient(to bottom, rgba(8,8,12,0.3) 0%, rgba(8,8,12,0.85) 50%, rgba(8,8,12,0.98) 100%)",
          zIndex: 1,
        }}
      />

      {/* Main content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          className="flex items-end justify-between mb-10"
        >
          <div>
            <h1 className="text-4xl font-bold font-display tracking-tight flex items-center gap-3">
              <span className="text-gradient-primary">Dashboard</span>
              <Sparkles size={24} className="text-violet-400" />
            </h1>
            <p className="text-muted-foreground mt-2">
              Welcome back. Your annotation workspace awaits.
            </p>
          </div>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate("/studio")}
            className="hidden md:flex items-center gap-2 px-5 py-2.5 btn-primary rounded-lg font-medium"
          >
            Open Studio <ArrowRight size={16} />
          </motion.button>
        </motion.header>

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 flex items-center gap-3"
            >
              <AlertCircle size={18} className="text-amber-400 shrink-0" />
              <p className="text-sm text-amber-300">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bento Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Stats row - top */}
          <div className="lg:col-span-12 grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              icon={FolderOpen}
              label="Total Projects"
              value={isLoading ? "—" : stats.totalProjects}
              color="violet"
              onClick={() => navigate("/studio")}
              delay={0}
            />
            <StatCard
              icon={Layers}
              label="Total Assets"
              value={isLoading ? "—" : stats.totalAssets}
              trend={stats.totalAssets > 0 ? "+12%" : undefined}
              color="emerald"
              onClick={() => navigate("/studio")}
              delay={1}
            />
            <StatCard
              icon={Tag}
              label="Annotations"
              value={isLoading ? "—" : stats.totalAnnotations}
              color="sky"
              delay={2}
            />
            <StatCard
              icon={Activity}
              label="Today's Activity"
              value={isLoading ? "—" : stats.todayAnnotations}
              color="amber"
              delay={3}
            />
          </div>

          {/* Main content - upload zone (left) */}
          <div className="lg:col-span-8">
            <HeroUpload
              onFilesSelected={handleFilesSelected}
              isUploading={isUploading}
              uploadStatus={uploadStatus}
            />
            {/* Dataset URL Import */}
            <DatasetImport onImport={handleDatasetImport} />
          </div>

          {/* Sidebar - quick actions (right) */}
          <div className="lg:col-span-4">
            <QuickActions onNavigate={navigate} />
          </div>

          {/* Bottom row - System Status and Recent Activity */}
          <div className="lg:col-span-5">
            <SystemStatus isLoading={isLoading} />
          </div>

          <div className="lg:col-span-7">
            <RecentActivity
              items={stats.recentActivity}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* SAM3 Features footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-10 pt-8 border-t border-white/[0.04]"
        >
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-muted-foreground">
            <span className="px-3 py-1.5 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20">
              SAM3 • 270K+ Concepts
            </span>
            <span className="px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
              RAG Intelligence
            </span>
            <span className="px-3 py-1.5 rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/20">
              LLM Auto-Prompts
            </span>
            <span className="px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
              Active Learning
            </span>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Home;
