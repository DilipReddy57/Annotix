import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import IntroScreen from "./components/IntroScreen";
import Layout from "./components/Layout";
import Home from "./components/Home";
import ProjectView from "./components/Project/ProjectView";
import SettingsPage from "./components/Settings";
import Analytics from "./components/Analytics";

function App() {
  const [showIntro, setShowIntro] = useState(true);

  // Handle intro completion
  const handleIntroComplete = () => {
    setShowIntro(false);
  };

  return (
    <>
      {/* Matrix Intro Screen */}
      {showIntro && <IntroScreen onComplete={handleIntroComplete} />}

      {/* Main App (hidden during intro) */}
      <div style={{ display: showIntro ? "none" : "block" }}>
        <Router>
          <Routes>
            {/* Dashboard */}
            <Route
              path="/"
              element={
                <Layout>
                  <Home />
                </Layout>
              }
            />

            {/* Studio (Gallery + Editor) */}
            <Route
              path="/studio"
              element={
                <Layout>
                  <ProjectView />
                </Layout>
              }
            />

            {/* Analytics */}
            <Route
              path="/analytics/:projectId"
              element={
                <Layout>
                  <Analytics />
                </Layout>
              }
            />

            {/* Settings */}
            <Route
              path="/settings"
              element={
                <Layout>
                  <SettingsPage />
                </Layout>
              }
            />

            {/* Datasets (Placeholder) */}
            <Route
              path="/datasets"
              element={
                <Layout>
                  <div className="p-8 text-center">
                    <h1 className="text-2xl font-bold mb-4">Datasets</h1>
                    <p className="text-muted-foreground">
                      Export and manage your annotation datasets. Coming soon.
                    </p>
                  </div>
                </Layout>
              }
            />

            {/* Redirects */}
            <Route
              path="/projects"
              element={<Navigate to="/studio" replace />}
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Router>
      </div>
    </>
  );
}

export default App;
