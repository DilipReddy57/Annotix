import { useState } from "react";
import {
  Settings,
  Bell,
  Search,
  ChevronDown,
  Home,
  LayoutGrid,
  Database,
  Menu,
  X,
  User,
  LogOut,
} from "lucide-react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import AnnotixLogo from "./ui/annotix-logo";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { icon: Home, label: "Dashboard", path: "/" },
    { icon: LayoutGrid, label: "Studio", path: "/studio" },
    { icon: Database, label: "Datasets", path: "/datasets" },
    { icon: Settings, label: "Settings", path: "/settings" },
  ];

  const isActive = (path: string) => {
    if (path === "/") return location.pathname === "/";
    return location.pathname.startsWith(path);
  };

  // Mock notifications
  const notifications = [
    { id: 1, title: "New annotation completed", time: "2 min ago" },
    { id: 2, title: "Project exported successfully", time: "1 hour ago" },
    { id: 3, title: "Welcome to ANNOTIX!", time: "Today" },
  ];

  return (
    <div className="min-h-screen w-full bg-background text-foreground flex flex-col">
      {/* Top Navigation */}
      <header className="h-16 border-b border-white/5 bg-background/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="h-full max-w-[1600px] mx-auto px-4 md:px-6 flex items-center justify-between">
          {/* Left: Logo + Nav */}
          <div className="flex items-center gap-8">
            {/* Logo */}
            <motion.div
              className="flex items-center cursor-pointer"
              onClick={() => navigate("/")}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <AnnotixLogo size="sm" />
            </motion.div>

            {/* Desktop Nav */}
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className={cn(
                    "relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                    isActive(item.path)
                      ? "text-white"
                      : "text-muted-foreground hover:text-white hover:bg-white/5"
                  )}
                >
                  <item.icon size={16} />
                  <span>{item.label}</span>
                  {isActive(item.path) && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 bg-white/10 rounded-lg -z-10"
                      transition={{
                        type: "spring",
                        bounce: 0.2,
                        duration: 0.6,
                      }}
                    />
                  )}
                </button>
              ))}
            </nav>
          </div>

          {/* Right: Search + Actions */}
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="hidden lg:flex items-center">
              <div className="relative">
                <Search
                  size={14}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
                />
                <input
                  type="text"
                  placeholder="Search..."
                  className="w-48 bg-white/5 border border-white/10 rounded-lg py-2 pl-9 pr-4 text-sm placeholder:text-muted-foreground focus:outline-none focus:border-primary/50 transition-colors"
                />
              </div>
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => {
                  setNotificationsOpen(!notificationsOpen);
                  setUserMenuOpen(false);
                }}
                className="relative p-2 rounded-lg text-muted-foreground hover:text-white hover:bg-white/5 transition-colors"
              >
                <Bell size={18} />
                <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full" />
              </button>

              <AnimatePresence>
                {notificationsOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute right-0 top-12 w-80 glass rounded-xl shadow-2xl overflow-hidden z-50"
                  >
                    <div className="p-4 border-b border-white/5">
                      <h3 className="font-semibold">Notifications</h3>
                    </div>
                    <div className="max-h-64 overflow-y-auto">
                      {notifications.map((notif) => (
                        <div
                          key={notif.id}
                          className="p-4 hover:bg-white/5 cursor-pointer transition-colors border-b border-white/5 last:border-0"
                        >
                          <p className="text-sm font-medium">{notif.title}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            {notif.time}
                          </p>
                        </div>
                      ))}
                    </div>
                    <div className="p-3 border-t border-white/5">
                      <button className="w-full text-center text-sm text-primary hover:underline">
                        View all notifications
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Settings (Desktop) */}
            <button
              onClick={() => navigate("/settings")}
              className="hidden md:flex p-2 rounded-lg text-muted-foreground hover:text-white hover:bg-white/5 transition-colors"
            >
              <Settings size={18} />
            </button>

            {/* Divider */}
            <div className="h-6 w-px bg-white/10 hidden sm:block mx-1" />

            {/* User Menu */}
            <div className="relative">
              <button
                onClick={() => {
                  setUserMenuOpen(!userMenuOpen);
                  setNotificationsOpen(false);
                }}
                className="hidden sm:flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
              >
                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center text-[10px] font-bold text-white">
                  U
                </div>
                <span className="text-sm font-medium">User</span>
                <ChevronDown size={14} className="text-muted-foreground" />
              </button>

              <AnimatePresence>
                {userMenuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute right-0 top-12 w-56 glass rounded-xl shadow-2xl overflow-hidden z-50"
                  >
                    <div className="p-4 border-b border-white/5">
                      <p className="font-semibold">User</p>
                      <p className="text-sm text-muted-foreground">
                        user@annotix.ai
                      </p>
                    </div>
                    <div className="p-2">
                      <button
                        onClick={() => {
                          navigate("/settings");
                          setUserMenuOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors text-left"
                      >
                        <User size={16} className="text-muted-foreground" />
                        <span className="text-sm">Profile</span>
                      </button>
                      <button
                        onClick={() => {
                          navigate("/settings");
                          setUserMenuOpen(false);
                        }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors text-left"
                      >
                        <Settings size={16} className="text-muted-foreground" />
                        <span className="text-sm">Settings</span>
                      </button>
                      <div className="my-2 border-t border-white/5" />
                      <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors text-left text-destructive">
                        <LogOut size={16} />
                        <span className="text-sm">Log out</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Mobile Menu Toggle */}
            <button
              className="md:hidden p-2 rounded-lg text-muted-foreground hover:text-white"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden border-b border-white/5 bg-background"
          >
            <nav className="p-4 space-y-2">
              {navItems.map((item) => (
                <button
                  key={item.path}
                  onClick={() => {
                    navigate(item.path);
                    setMobileMenuOpen(false);
                  }}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all",
                    isActive(item.path)
                      ? "bg-white/10 text-white"
                      : "text-muted-foreground hover:bg-white/5"
                  )}
                >
                  <item.icon size={18} />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click outside to close dropdowns */}
      {(userMenuOpen || notificationsOpen) && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => {
            setUserMenuOpen(false);
            setNotificationsOpen(false);
          }}
        />
      )}

      {/* Main Content */}
      <main className="flex-1">{children}</main>

      {/* Footer */}
      <footer className="border-t border-white/5 bg-background/50 backdrop-blur-sm py-4">
        <div className="max-w-[1600px] mx-auto px-4 md:px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            {/* Left: Status Indicators */}
            <div className="flex items-center gap-4">
              {/* SAM3 Status */}
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs font-mono text-muted-foreground">
                  SAM3 Model Ready
                </span>
              </div>

              {/* Tech Stack Badges */}
              <div className="hidden sm:flex items-center gap-2">
                <span className="px-2 py-0.5 text-[10px] font-mono bg-purple-500/20 border border-purple-500/30 rounded text-purple-300">
                  SAM3
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono bg-blue-500/20 border border-blue-500/30 rounded text-blue-300">
                  RAG
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono bg-cyan-500/20 border border-cyan-500/30 rounded text-cyan-300">
                  LLM
                </span>
                <span className="px-2 py-0.5 text-[10px] font-mono bg-green-500/20 border border-green-500/30 rounded text-green-300">
                  Active Learning
                </span>
              </div>
            </div>

            {/* Center: Version */}
            <div className="text-xs text-muted-foreground/60 font-mono">
              v1.0.0 • 270K+ Concepts
            </div>

            {/* Right: Creator Credit */}
            <div className="text-xs text-muted-foreground/60 font-mono">
              Created by{" "}
              <span className="text-primary hover:underline cursor-pointer">
                Dilip Reddy
              </span>{" "}
              • Built with <span className="text-accent">Antigravity</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
