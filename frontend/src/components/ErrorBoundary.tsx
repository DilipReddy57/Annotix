import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-black text-white p-8">
          <div className="max-w-2xl w-full bg-red-900/20 border border-red-500 rounded-2xl p-8">
            <h1 className="text-3xl font-bold text-red-500 mb-4">
              Application Crashed
            </h1>
            <p className="text-xl mb-4">
              Something went wrong in the frontend.
            </p>
            <div className="bg-black/50 p-4 rounded-lg overflow-auto font-mono text-sm max-h-96">
              <p className="text-red-300 font-bold">
                {this.state.error?.toString()}
              </p>
              <br />
              <p className="text-gray-400">
                Please report this to the developer.
              </p>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="mt-6 bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
