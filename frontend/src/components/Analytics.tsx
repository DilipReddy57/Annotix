"use client";

import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { motion } from "framer-motion";
import {
  BarChart3,
  PieChart,
  TrendingUp,
  Target,
  ArrowLeft,
  Download,
  Loader2,
} from "lucide-react";
import { API_BASE_URL } from "../api/client";

interface AnalyticsData {
  project_id: string;
  project_name: string;
  total_images: number;
  total_annotations: number;
  class_distribution: Array<{ name: string; count: number }>;
  confidence_distribution: Array<{ range: string; count: number }>;
  status_breakdown: Array<{ status: string; count: number }>;
  avg_annotations_per_image: number;
}

const Analytics = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      if (!projectId) return;

      try {
        setIsLoading(true);
        const response = await axios.get(
          `${API_BASE_URL}/api/export/${projectId}/analytics`
        );
        setAnalytics(response.data);
      } catch (err: any) {
        setError(err.message || "Failed to load analytics");
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalytics();
  }, [projectId]);

  const handleExportCOCO = async () => {
    if (!projectId) return;
    window.open(
      `${API_BASE_URL}/api/export/${projectId}/export/coco`,
      "_blank"
    );
  };

  const handleExportYOLO = async () => {
    if (!projectId) return;
    window.open(
      `${API_BASE_URL}/api/export/${projectId}/export/yolo`,
      "_blank"
    );
  };

  // Color palette for charts
  const colors = [
    "#10b981",
    "#14b8a6",
    "#06b6d4",
    "#3b82f6",
    "#8b5cf6",
    "#a855f7",
    "#d946ef",
    "#ec4899",
    "#f43f5e",
    "#f97316",
  ];

  const statusColors: Record<string, string> = {
    pending: "#f59e0b",
    processing: "#3b82f6",
    completed: "#10b981",
    error: "#ef4444",
    rejected: "#6b7280",
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <p className="text-red-400">{error || "No data available"}</p>
        <button
          onClick={() => navigate(-1)}
          className="btn-ghost flex items-center gap-2"
        >
          <ArrowLeft size={16} />
          Go Back
        </button>
      </div>
    );
  }

  const maxClassCount = Math.max(
    ...analytics.class_distribution.map((c) => c.count),
    1
  );
  const maxConfCount = Math.max(
    ...analytics.confidence_distribution.map((c) => c.count),
    1
  );

  return (
    <div className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-8"
      >
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
          >
            <ArrowLeft size={20} />
          </button>
          <div>
            <h1 className="text-2xl font-bold font-display">
              Analytics: {analytics.project_name}
            </h1>
            <p className="text-muted-foreground text-sm">
              {analytics.total_images} images • {analytics.total_annotations}{" "}
              annotations
            </p>
          </div>
        </div>

        {/* Export buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleExportCOCO}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <Download size={16} />
            COCO
          </button>
          <button
            onClick={handleExportYOLO}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            <Download size={16} />
            YOLO
          </button>
        </div>
      </motion.header>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {[
          {
            label: "Total Images",
            value: analytics.total_images,
            icon: PieChart,
            color: "emerald",
          },
          {
            label: "Annotations",
            value: analytics.total_annotations,
            icon: Target,
            color: "teal",
          },
          {
            label: "Classes",
            value: analytics.class_distribution.length,
            icon: BarChart3,
            color: "cyan",
          },
          {
            label: "Avg/Image",
            value: analytics.avg_annotations_per_image,
            icon: TrendingUp,
            color: "blue",
          },
        ].map((stat, i) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-card rounded-xl p-4 border border-white/5"
          >
            <div className="flex items-center gap-3 mb-2">
              <stat.icon size={18} className="text-primary" />
              <span className="text-muted-foreground text-sm">
                {stat.label}
              </span>
            </div>
            <p className="text-2xl font-bold">{stat.value}</p>
          </motion.div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Class Distribution */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-card rounded-xl p-6 border border-white/5"
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <BarChart3 size={18} className="text-primary" />
            Class Distribution
          </h3>
          <div className="space-y-3">
            {analytics.class_distribution.slice(0, 10).map((cls, i) => (
              <div key={cls.name} className="flex items-center gap-3">
                <span className="w-24 text-sm text-muted-foreground truncate">
                  {cls.name}
                </span>
                <div className="flex-1 h-6 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${(cls.count / maxClassCount) * 100}%` }}
                    transition={{ delay: 0.3 + i * 0.05, duration: 0.5 }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: colors[i % colors.length] }}
                  />
                </div>
                <span className="w-12 text-sm text-right">{cls.count}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Confidence Distribution */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-card rounded-xl p-6 border border-white/5"
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp size={18} className="text-primary" />
            Confidence Distribution
          </h3>
          <div className="flex items-end gap-2 h-40">
            {analytics.confidence_distribution.map((bucket, i) => (
              <div
                key={bucket.range}
                className="flex-1 flex flex-col items-center gap-2"
              >
                <motion.div
                  initial={{ height: 0 }}
                  animate={{
                    height: `${(bucket.count / maxConfCount) * 100}%`,
                  }}
                  transition={{ delay: 0.4 + i * 0.1, duration: 0.5 }}
                  className="w-full rounded-t-lg min-h-[4px]"
                  style={{ backgroundColor: colors[i + 2] }}
                />
                <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                  {bucket.range}
                </span>
                <span className="text-xs font-medium">{bucket.count}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Status Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-card rounded-xl p-6 border border-white/5 md:col-span-2"
        >
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <PieChart size={18} className="text-primary" />
            Status Breakdown
          </h3>
          <div className="flex flex-wrap gap-4">
            {analytics.status_breakdown.map((status) => (
              <div
                key={status.status}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5"
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor: statusColors[status.status] || "#6b7280",
                  }}
                />
                <span className="text-sm capitalize">{status.status}</span>
                <span className="text-sm font-bold">{status.count}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Analytics;
