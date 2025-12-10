import React from "react";
import ProjectView from "./Project/ProjectView";

// The "Dashboard" is now just the Studio (Gallery) View
// This removes the "Middle World" / Globe page entirely.
const Dashboard = () => {
  return (
    <div className="h-full flex flex-col">
      <ProjectView />
    </div>
  );
};

export default Dashboard;
