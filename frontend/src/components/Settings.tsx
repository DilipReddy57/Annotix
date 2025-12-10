"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Settings,
  Bell,
  Palette,
  Database,
  Shield,
  Monitor,
  Moon,
  Sun,
  ChevronRight,
  Check,
  Trash2,
  Download,
  Upload,
  Save,
} from "lucide-react";
import { cn } from "@/lib/utils";

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
};

interface SettingSection {
  id: string;
  icon: any;
  title: string;
  description: string;
}

const sections: SettingSection[] = [
  {
    id: "sam3",
    icon: Settings,
    title: "SAM3 Model",
    description: "Model configuration and inference settings",
  },
  {
    id: "appearance",
    icon: Palette,
    title: "Appearance",
    description: "Theme, colors, and display",
  },
  {
    id: "notifications",
    icon: Bell,
    title: "Notifications",
    description: "Alerts and updates",
  },
  {
    id: "data",
    icon: Database,
    title: "Data & Storage",
    description: "Export, import, and cleanup",
  },
  {
    id: "display",
    icon: Monitor,
    title: "Display",
    description: "Resolution and performance",
  },
  {
    id: "privacy",
    icon: Shield,
    title: "Privacy",
    description: "Data handling and security",
  },
];

const SettingsPage = () => {
  const [activeSection, setActiveSection] = useState("sam3");
  const [theme, setTheme] = useState<"dark" | "light" | "system">("dark");
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    sounds: true,
    annotations: true,
  });
  const [saved, setSaved] = useState(false);
  const [sam3Config, setSam3Config] = useState({
    confidenceThreshold: 0.5,
    maxAnnotations: 100,
    useAutoPrompts: true,
    enableRAG: true,
    enableActiveLearning: true,
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const renderContent = () => {
    switch (activeSection) {
      case "sam3":
        return (
          <div className="space-y-6">
            {/* Model Status */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                <div>
                  <p className="font-medium text-green-400">SAM3 Model Ready</p>
                  <p className="text-sm text-muted-foreground">
                    270,000+ concepts available
                  </p>
                </div>
              </div>
            </div>

            {/* Confidence Threshold */}
            <div>
              <h3 className="text-lg font-semibold mb-4">
                Confidence Threshold
              </h3>
              <div className="space-y-3">
                <input
                  type="range"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={sam3Config.confidenceThreshold}
                  onChange={(e) =>
                    setSam3Config((c) => ({
                      ...c,
                      confidenceThreshold: parseFloat(e.target.value),
                    }))
                  }
                  className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>Low (0.1)</span>
                  <span className="text-primary font-medium">
                    {sam3Config.confidenceThreshold}
                  </span>
                  <span>High (0.95)</span>
                </div>
              </div>
            </div>

            {/* Feature Toggles */}
            <div>
              <h3 className="text-lg font-semibold mb-4">AI Features</h3>
              <div className="space-y-3">
                {[
                  {
                    key: "useAutoPrompts",
                    label: "LLM Auto-Prompts",
                    desc: "Generate prompts using Gemini/GPT",
                  },
                  {
                    key: "enableRAG",
                    label: "RAG Intelligence",
                    desc: "Label consistency with knowledge base",
                  },
                  {
                    key: "enableActiveLearning",
                    label: "Active Learning",
                    desc: "Smart sample selection",
                  },
                ].map((item) => (
                  <div
                    key={item.key}
                    className="flex items-center justify-between p-4 rounded-xl border border-white/10 hover:border-white/20 transition-all"
                  >
                    <div>
                      <p className="font-medium">{item.label}</p>
                      <p className="text-sm text-muted-foreground">
                        {item.desc}
                      </p>
                    </div>
                    <button
                      onClick={() =>
                        setSam3Config((c) => ({
                          ...c,
                          [item.key]: !c[item.key as keyof typeof c],
                        }))
                      }
                      className={cn(
                        "w-12 h-6 rounded-full transition-all relative",
                        sam3Config[item.key as keyof typeof sam3Config]
                          ? "bg-primary"
                          : "bg-white/20"
                      )}
                    >
                      <div
                        className={cn(
                          "w-5 h-5 rounded-full bg-white absolute top-0.5 transition-all",
                          sam3Config[item.key as keyof typeof sam3Config]
                            ? "left-6"
                            : "left-0.5"
                        )}
                      />
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {/* API Keys */}
            <div>
              <h3 className="text-lg font-semibold mb-4">API Configuration</h3>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">
                    Gemini API Key
                  </label>
                  <input
                    type="password"
                    placeholder="Enter your Gemini API key"
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-primary/50"
                  />
                </div>
                <div>
                  <label className="block text-sm text-muted-foreground mb-2">
                    Hugging Face Token
                  </label>
                  <input
                    type="password"
                    placeholder="Enter your HuggingFace token"
                    className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:border-primary/50"
                  />
                </div>
              </div>
            </div>
          </div>
        );

      case "appearance":
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Theme</h3>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { id: "dark", icon: Moon, label: "Dark" },
                  { id: "light", icon: Sun, label: "Light" },
                  { id: "system", icon: Monitor, label: "System" },
                ].map((option) => (
                  <button
                    key={option.id}
                    onClick={() => setTheme(option.id as any)}
                    className={cn(
                      "p-4 rounded-xl border flex flex-col items-center gap-2 transition-all",
                      theme === option.id
                        ? "border-primary bg-primary/10 text-primary"
                        : "border-white/10 hover:border-white/20"
                    )}
                  >
                    <option.icon size={24} />
                    <span className="text-sm font-medium">{option.label}</span>
                    {theme === option.id && (
                      <Check size={16} className="text-primary" />
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">Accent Color</h3>
              <div className="flex gap-3">
                {["#8b5cf6", "#06b6d4", "#22c55e", "#f59e0b", "#ef4444"].map(
                  (color) => (
                    <button
                      key={color}
                      className="w-10 h-10 rounded-full border-2 border-white/20 hover:border-white/50 transition-all hover:scale-110"
                      style={{ backgroundColor: color }}
                    />
                  )
                )}
              </div>
            </div>
          </div>
        );

      case "notifications":
        return (
          <div className="space-y-4">
            {[
              {
                key: "email",
                label: "Email Notifications",
                desc: "Receive updates via email",
              },
              {
                key: "push",
                label: "Push Notifications",
                desc: "Browser push alerts",
              },
              {
                key: "sounds",
                label: "Sound Effects",
                desc: "Play sounds for actions",
              },
              {
                key: "annotations",
                label: "Annotation Alerts",
                desc: "Notify on new annotations",
              },
            ].map((item) => (
              <div
                key={item.key}
                className="flex items-center justify-between p-4 rounded-xl border border-white/10 hover:border-white/20 transition-all"
              >
                <div>
                  <p className="font-medium">{item.label}</p>
                  <p className="text-sm text-muted-foreground">{item.desc}</p>
                </div>
                <button
                  onClick={() =>
                    setNotifications((n) => ({
                      ...n,
                      [item.key]: !n[item.key as keyof typeof n],
                    }))
                  }
                  className={cn(
                    "w-12 h-6 rounded-full transition-all relative",
                    notifications[item.key as keyof typeof notifications]
                      ? "bg-primary"
                      : "bg-white/20"
                  )}
                >
                  <div
                    className={cn(
                      "w-5 h-5 rounded-full bg-white absolute top-0.5 transition-all",
                      notifications[item.key as keyof typeof notifications]
                        ? "left-6"
                        : "left-0.5"
                    )}
                  />
                </button>
              </div>
            ))}
          </div>
        );

      case "data":
        return (
          <div className="space-y-4">
            <button className="w-full flex items-center justify-between p-4 rounded-xl border border-white/10 hover:border-white/20 transition-all group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center text-accent">
                  <Download size={20} />
                </div>
                <div className="text-left">
                  <p className="font-medium">Export All Data</p>
                  <p className="text-sm text-muted-foreground">
                    Download all projects as ZIP
                  </p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-muted-foreground group-hover:text-white"
              />
            </button>

            <button className="w-full flex items-center justify-between p-4 rounded-xl border border-white/10 hover:border-white/20 transition-all group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                  <Upload size={20} />
                </div>
                <div className="text-left">
                  <p className="font-medium">Import Data</p>
                  <p className="text-sm text-muted-foreground">
                    Restore from backup
                  </p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-muted-foreground group-hover:text-white"
              />
            </button>

            <button className="w-full flex items-center justify-between p-4 rounded-xl border border-destructive/20 hover:border-destructive/50 transition-all group">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-lg bg-destructive/10 flex items-center justify-center text-destructive">
                  <Trash2 size={20} />
                </div>
                <div className="text-left">
                  <p className="font-medium text-destructive">Clear All Data</p>
                  <p className="text-sm text-muted-foreground">
                    Delete all projects and annotations
                  </p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-destructive/50 group-hover:text-destructive"
              />
            </button>
          </div>
        );

      default:
        return (
          <div className="text-center py-12 text-muted-foreground">
            <Settings size={48} className="mx-auto mb-4 opacity-50" />
            <p>Settings for {activeSection} coming soon.</p>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen p-6 md:p-8">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={stagger}
        className="max-w-5xl mx-auto"
      >
        {/* Header */}
        <motion.div variants={fadeIn} className="mb-8">
          <h1 className="text-3xl font-bold font-display flex items-center gap-3">
            <Settings size={28} className="text-primary" />
            Settings
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage your workspace preferences and configuration.
          </p>
        </motion.div>

        {/* Layout */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Sidebar */}
          <motion.div variants={fadeIn} className="space-y-2">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={cn(
                  "w-full flex items-center gap-3 p-3 rounded-xl text-left transition-all",
                  activeSection === section.id
                    ? "bg-primary/10 text-primary border border-primary/20"
                    : "hover:bg-white/5 text-muted-foreground hover:text-white"
                )}
              >
                <section.icon size={18} />
                <span className="font-medium text-sm">{section.title}</span>
              </button>
            ))}
          </motion.div>

          {/* Content */}
          <motion.div variants={fadeIn} className="md:col-span-3">
            <div className="glass rounded-2xl p-6">
              <div className="mb-6">
                <h2 className="text-xl font-semibold">
                  {sections.find((s) => s.id === activeSection)?.title}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {sections.find((s) => s.id === activeSection)?.description}
                </p>
              </div>

              {renderContent()}

              {/* Save Button */}
              <div className="mt-8 pt-6 border-t border-white/5 flex justify-end">
                <button
                  onClick={handleSave}
                  className={cn(
                    "flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-all",
                    saved
                      ? "bg-success text-success-foreground"
                      : "bg-primary text-primary-foreground hover:bg-primary/90"
                  )}
                >
                  {saved ? (
                    <>
                      <Check size={16} /> Saved
                    </>
                  ) : (
                    <>
                      <Save size={16} /> Save Changes
                    </>
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default SettingsPage;
