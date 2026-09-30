import React from "react";
import { Radio, RefreshCw, Wrench, Settings, LayoutDashboard, Bike } from "lucide-react";

export default function Navbar({
  connected,
  isSimulated,
  status,
  activeTab = "dashboard",
  onTabChange,
  onOpenSettings,
  onOpenDiagnostic,
  onResetMetrics,
  isResetting,
}) {
  let badgeClasses = "bg-zinc-100 text-zinc-600 border-zinc-200";
  let dotClasses = "bg-zinc-400";
  let label = "DISCONNECTED";

  if (status === "connecting") {
    badgeClasses = "bg-amber-50 text-amber-800 border-amber-200";
    dotClasses = "bg-amber-500 animate-pulse";
    label = "CONNECTING...";
  } else if (connected) {
    if (isSimulated) {
      badgeClasses = "bg-zinc-100 text-zinc-800 border-zinc-300";
      dotClasses = "bg-zinc-700";
      label = "SIMULATION ACTIVE";
    } else {
      badgeClasses = "bg-emerald-50 text-emerald-800 border-emerald-200";
      dotClasses = "bg-emerald-600";
      label = "CONNECTED";
    }
  }

  return (
    <header className="border-b border-zinc-200 bg-white sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-md bg-zinc-900 text-white flex items-center justify-center">
            <Radio className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-semibold text-zinc-950 tracking-tight">
                TF02-Pro LiDAR Pothole Detection
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2.5">
          {/* Connection status indicator */}
          <div
            className={`inline-flex items-center space-x-2 px-2.5 py-1 rounded-md text-xs font-medium border ${badgeClasses}`}
          >
            <span className={`w-2 h-2 rounded-full ${dotClasses}`}></span>
            <span className="font-mono text-[11px] font-semibold tracking-wider">
              {label}
            </span>
          </div>

          <div className="h-4 w-px bg-zinc-200 mx-1"></div>

          {/* Action buttons */}
          <button
            onClick={onResetMetrics}
            disabled={isResetting}
            title="Reset telemetry metrics and event logs"
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-zinc-50 border border-zinc-300 transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isResetting ? "animate-spin" : ""}`} />
            <span className="hidden sm:inline">Reset</span>
          </button>

          <button
            onClick={onOpenDiagnostic}
            title="Open hardware diagnostic panel"
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-zinc-50 border border-zinc-300 transition"
          >
            <Wrench className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Diagnostics</span>
          </button>

          <button
            onClick={onOpenSettings}
            title="Open detection thresholds and vehicle settings"
            className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-zinc-50 border border-zinc-300 transition"
          >
            <Settings className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Settings</span>
          </button>
        </div>
      </div>
    </header>
  );
}
