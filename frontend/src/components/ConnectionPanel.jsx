import React from "react";
import {
  Wifi,
  WifiOff,
  RefreshCw,
  Play,
  Square,
  Cpu,
} from "lucide-react";

export default function ConnectionPanel({
  connected,
  isSimulated,
  status,
  statusMessage,
  selectedPort,
  setSelectedPort,
  availablePorts,
  onRefreshPorts,
  baudRate,
  setBaudRate,
  onConnect,
  onDisconnect,
  onToggleSimulate,
  framesReceived,
  errorsCount,
  isConnecting,
}) {
  const isOnline = connected;

  let statusBadgeClasses = "bg-zinc-100 text-zinc-700 border-zinc-200";
  let statusText = "Disconnected";
  let iconContainer = "bg-zinc-100 border-zinc-200 text-zinc-500";

  if (status === "connecting") {
    statusBadgeClasses = "bg-amber-50 text-amber-800 border-amber-200";
    statusText = "Connecting...";
    iconContainer = "bg-amber-50 border-amber-200 text-amber-700";
  } else if (isOnline) {
    if (isSimulated) {
      statusBadgeClasses = "bg-zinc-100 text-zinc-900 border-zinc-300";
      statusText = "Simulated Link";
      iconContainer = "bg-zinc-100 border-zinc-300 text-zinc-800";
    } else {
      statusBadgeClasses = "bg-emerald-50 text-emerald-800 border-emerald-200";
      statusText = "Hardware Connected";
      iconContainer = "bg-emerald-50 border-emerald-200 text-emerald-700";
    }
  } else if (status === "error") {
    statusBadgeClasses = "bg-rose-50 text-rose-800 border-rose-200";
    statusText = "Error";
    iconContainer = "bg-rose-50 border-rose-200 text-rose-700";
  }

  return (
    <div className="bg-white border border-zinc-200 rounded-lg p-5">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5">
        {/* Left: Connection State */}
        <div className="flex items-start sm:items-center space-x-4">
          <div
            className={`w-11 h-11 rounded-md flex items-center justify-center border transition-colors ${iconContainer}`}
          >
            {isOnline ? (
              <Wifi className="w-5 h-5" />
            ) : status === "connecting" ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <WifiOff className="w-5 h-5" />
            )}
          </div>

          <div>
            <div className="flex items-center space-x-2">
              <span className="text-[11px] uppercase tracking-wider font-semibold text-zinc-500">
                LiDAR Interface
              </span>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold border ${statusBadgeClasses}`}
              >
                {statusText}
              </span>
            </div>

            <p className="text-sm font-medium text-zinc-900 mt-1">
              {statusMessage ||
                (isOnline
                  ? `Active on ${selectedPort} at ${baudRate} baud`
                  : "Sensor not connected. Select a port or start simulation.")}
            </p>

            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500 mt-1.5 font-mono">
              <span>
                Thread: <strong className="text-zinc-800 font-semibold">100 Hz</strong>
              </span>
              <span className="text-zinc-300">|</span>
              <span>
                Telemetry: <strong className="text-zinc-800 font-semibold">10 Hz</strong>
              </span>
              <span className="text-zinc-300">|</span>
              <span>
                Frames: <strong className="text-zinc-800 font-semibold">{framesReceived.toLocaleString()}</strong>
              </span>
              <span className="text-zinc-300">|</span>
              <span>
                Errors:{" "}
                <strong
                  className={
                    errorsCount > 0
                      ? "text-rose-600 font-bold"
                      : "text-zinc-800 font-semibold"
                  }
                >
                  {errorsCount}
                </strong>
              </span>
            </div>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex flex-wrap items-center gap-2 pt-3 lg:pt-0 border-t lg:border-t-0 border-zinc-100">
          <div className="flex items-center space-x-1.5">
            <select
              disabled={isOnline || isConnecting}
              value={selectedPort}
              onChange={(e) => setSelectedPort(e.target.value)}
              className="bg-white text-zinc-900 text-xs rounded-md px-3 py-1.5 border border-zinc-300 focus:outline-none focus:border-zinc-500 disabled:opacity-50 font-mono"
            >
              {availablePorts.length > 0 ? (
                availablePorts.map((p) => (
                  <option key={p} value={p}>
                    {p === "auto" ? " Auto-Detect (Scan Ports)" : p}
                  </option>
                ))
              ) : (
                <>
                  <option value="auto"> Auto-Detect (Scan Ports)</option>
                  <option value={selectedPort}>{selectedPort} (Manual)</option>
                </>
              )}
            </select>

            <button
              onClick={onRefreshPorts}
              disabled={isOnline || isConnecting}
              title="Scan serial ports"
              className="p-1.5 rounded-md bg-white hover:bg-zinc-50 text-zinc-600 border border-zinc-300 transition disabled:opacity-50"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <select
            disabled={isOnline || isConnecting}
            value={baudRate}
            onChange={(e) => setBaudRate(Number(e.target.value))}
            className="bg-white text-zinc-900 text-xs rounded-md px-2.5 py-1.5 border border-zinc-300 focus:outline-none focus:border-zinc-500 disabled:opacity-50 font-mono"
          >
            <option value={115200}>115200 baud</option>
            <option value={9600}>9600 baud</option>
          </select>

          {isOnline ? (
            <button
              onClick={onDisconnect}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-rose-50 text-rose-700 border border-rose-200 hover:bg-rose-100 transition"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Disconnect</span>
            </button>
          ) : (
            <button
              onClick={() => onConnect(false)}
              disabled={isConnecting}
              className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-md text-xs font-semibold bg-zinc-900 hover:bg-zinc-800 text-white transition disabled:opacity-50 shadow-sm"
            >
              <Play className="w-3 h-3 fill-current" />
              <span>{isConnecting ? "Connecting..." : "Connect"}</span>
            </button>
          )}

          <button
            onClick={onToggleSimulate}
            className={`inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium border transition ${isSimulated
                ? "bg-zinc-900 text-white border-zinc-900"
                : "bg-white text-zinc-700 border-zinc-300 hover:bg-zinc-50"
              }`}
            title="Toggle simulated 100 Hz sensor stream"
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>{isSimulated ? "Stop Sim" : "Simulate"}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
