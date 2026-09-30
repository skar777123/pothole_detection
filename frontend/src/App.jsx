import React, { useState, useEffect, useRef } from "react";
import Navbar from "./components/Navbar";
import ConnectionPanel from "./components/ConnectionPanel";
import TelemetryCards from "./components/TelemetryCards";
import LiveCharts from "./components/LiveCharts";
import DetectionLog from "./components/DetectionLog";
import DiagnosticModal from "./components/DiagnosticModal";
import SettingsModal from "./components/SettingsModal";
import RoadSimulation2D from "./components/RoadSimulation2D";
import { Maximize2, X } from "lucide-react";

export default function App() {
  // Connection state
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("disconnected");
  const [statusMessage, setStatusMessage] = useState("Sensor not connected");
  const [selectedPort, setSelectedPort] = useState(() => {
    return localStorage.getItem("pothole_selected_port") || "auto";
  });
  const [availablePorts, setAvailablePorts] = useState([]);
  const [baudRate, setBaudRate] = useState(() => {
    const cached = localStorage.getItem("pothole_baud_rate");
    return cached ? Number(cached) : 115200;
  });
  const [framesReceived, setFramesReceived] = useState(0);
  const [errorsCount, setErrorsCount] = useState(0);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  // Mobile Canvas Fullscreen State
  const [isMobileCanvasOpen, setIsMobileCanvasOpen] = useState(false);

  // Reset trigger counter for simulation sync
  const [resetTrigger, setResetTrigger] = useState(0);

  // Telemetry metrics
  const [telemetry, setTelemetry] = useState({
    distance_cm: 0,
    baseline_cm: 0,
    deviation_cm: 0,
    strength: 0,
    temperature_c: 0,
    calibrated: false,
    warmup_count: 0,
    warmup_total: 20,
    class_id: 0,
    class_name: "Flat Road",
    confidence: 0,
    streak: 0,
    streak_target: 2,
    is_alert: false,
    alert_message: "",
    cooldown_remaining: 0,
    pi_temperature: null,
  });

  const [potholeCount, setPotholeCount] = useState(0);
  const [bumpCount, setBumpCount] = useState(0);
  const [lastDepth, setLastDepth] = useState(0);

  // History buffers for real-time charts
  const [history, setHistory] = useState({
    distance: [],
    deviation: [],
    strength: [],
    baseline: [],
  });

  // Detection log list loaded from cache
  const [logs, setLogs] = useState(() => {
    try {
      const cached = localStorage.getItem("pothole_logs");
      return cached ? JSON.parse(cached) : [];
    } catch {
      return [];
    }
  });

  // Settings loaded from cache
  const [settings, setSettings] = useState(() => {
    try {
      const cached = localStorage.getItem("pothole_settings");
      if (cached) return JSON.parse(cached);
    } catch {}
    return {
      speed_kmph: 30,
      pot_thresh: 4.5,
      deep_thresh: 8.0,
      bump_thresh: 4.5,
      confirm_n: 2,
      cooldown_s: 3.0,
    };
  });

  // Modal states
  const [isDiagnosticOpen, setIsDiagnosticOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // WebSocket reference
  const wsRef = useRef(null);

  // Fetch available COM / serial ports
  const fetchPorts = async () => {
    try {
      const res = await fetch("/api/ports");
      const data = await res.json();
      if (data && data.ports) {
        setAvailablePorts(data.ports);
        if (data.ports.length > 0 && !data.ports.includes(selectedPort)) {
          setSelectedPort(data.ports[0]);
        }
      }
    } catch {
      // Backend starting or offline
    }
  };

  // Fetch initial status
  const fetchInitialStatus = async () => {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data) {
        setConnected(data.connected);
        setStatus(data.status);
        setStatusMessage(data.status_message);
        if (data.port) setSelectedPort(data.port);
        if (data.baudrate) setBaudRate(data.baudrate);
        if (data.frames_received !== undefined) setFramesReceived(data.frames_received);
        if (data.errors_count !== undefined) setErrorsCount(data.errors_count);
        if (data.pothole_count !== undefined) setPotholeCount(data.pothole_count);
        if (data.bump_count !== undefined) setBumpCount(data.bump_count);
        if (data.last_depth !== undefined) setLastDepth(data.last_depth);
        if (data.telemetry) setTelemetry(data.telemetry);
        if (data.settings) setSettings(data.settings);
      }
    } catch {
      // Backend not yet reachable
    }
  };

  // Fetch detection log
  const fetchLog = async () => {
    try {
      const res = await fetch("/api/log");
      const data = await res.json();
      if (data && data.log) {
        setLogs(data.log);
      }
    } catch {
      // Backend not yet reachable
    }
  };

  // Connect to WebSocket stream
  useEffect(() => {
    fetchPorts();
    fetchInitialStatus();
    fetchLog();

    let ws = null;
    let reconnectTimeout = null;

    const connectWs = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = window.location.hostname || "localhost";
      const wsPort = window.location.port === "3000" ? "8000" : (window.location.port || (protocol === "wss:" ? "443" : "80"));
      const wsUrl = `${protocol}//${host}:${wsPort}/ws`;

      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setConnected(data.connected);
          setStatus(data.status);
          setStatusMessage(data.status_message);
          if (data.port) setSelectedPort(data.port);
          if (data.baudrate) setBaudRate(data.baudrate);
          setFramesReceived(data.frames_received || 0);
          setErrorsCount(data.errors_count || 0);
          setPotholeCount(data.pothole_count || 0);
          setBumpCount(data.bump_count || 0);
          setLastDepth(data.last_depth || 0);

          if (data.telemetry) {
            setTelemetry(data.telemetry);
          }

          if (data.history) {
            setHistory(data.history);
          }

          if (data.log_preview && data.log_preview.length > 0) {
            setLogs((prev) => {
              const existingIds = new Set(prev.map((item) => item.id));
              const newItems = data.log_preview.filter(
                (item) => !existingIds.has(item.id)
              );
              if (newItems.length > 0) {
                return [...newItems, ...prev];
              }
              return prev;
            });
          }
        } catch {
          // Parse error
        }
      };

      ws.onclose = () => {
        reconnectTimeout = setTimeout(connectWs, 2000);
      };

      ws.onerror = () => {
        if (ws) ws.close();
      };
    };

    connectWs();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  // Connect to hardware
  const handleConnect = async () => {
    setIsConnecting(true);
    setStatus("connecting");
    setStatusMessage(`Connecting to ${selectedPort} at ${baudRate} baud...`);
    try {
      const res = await fetch("/api/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          port: selectedPort,
          baudrate: baudRate,
        }),
      });
      const data = await res.json();
      setConnected(data.success);
      setStatus(data.success ? "connected" : "error");
      setStatusMessage(data.message);
    } catch (err) {
      setConnected(false);
      setStatus("error");
      setStatusMessage(`Connection failed: ${err.message}`);
    } finally {
      setIsConnecting(false);
    }
  };

  // Disconnect
  const handleDisconnect = async () => {
    try {
      const res = await fetch("/api/disconnect", { method: "POST" });
      const data = await res.json();
      setConnected(false);
      setStatus("disconnected");
      setStatusMessage(data.message || "Disconnected");
    } catch (err) {
      setStatusMessage(`Disconnect error: ${err.message}`);
    }
  };

  // Sync state to localStorage cache
  useEffect(() => {
    if (selectedPort) {
      try {
        localStorage.setItem("pothole_selected_port", selectedPort);
      } catch {}
    }
  }, [selectedPort]);

  useEffect(() => {
    if (baudRate) {
      try {
        localStorage.setItem("pothole_baud_rate", String(baudRate));
      } catch {}
    }
  }, [baudRate]);

  useEffect(() => {
    if (logs && logs.length > 0) {
      try {
        localStorage.setItem("pothole_logs", JSON.stringify(logs.slice(0, 100)));
      } catch {}
    }
  }, [logs]);

  // Reset metrics and clear detection cache
  const handleResetMetrics = async () => {
    setIsResetting(true);
    try {
      await fetch("/api/reset", { method: "POST" });
      setPotholeCount(0);
      setBumpCount(0);
      setLastDepth(0);
      setLogs([]);
      setResetTrigger((prev) => prev + 1);
      try {
        localStorage.removeItem("pothole_logs");
      } catch {}
      setHistory({
        distance: [],
        deviation: [],
        strength: [],
        baseline: [],
      });
    } catch {
      // Error
    } finally {
      setIsResetting(false);
    }
  };

  // Save settings and persist in cache
  const handleSaveSettings = async (newSettings) => {
    try {
      await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSettings),
      });
      setSettings(newSettings);
      try {
        localStorage.setItem("pothole_settings", JSON.stringify(newSettings));
      } catch {}
    } catch {
      // Error
    }
  };

  // Clear logs and purge cache
  const handleClearLog = () => {
    setLogs([]);
    try {
      localStorage.removeItem("pothole_logs");
    } catch {}
  };

  // Fullscreen open/close logic
  const openCanvasFullscreen = async () => {
    setIsMobileCanvasOpen(true);
    try {
      const el = document.documentElement;
      if (el.requestFullscreen) {
        await el.requestFullscreen();
      }
      if (window.screen && window.screen.orientation && window.screen.orientation.lock) {
        await window.screen.orientation.lock("landscape");
      }
    } catch (err) {
      console.warn("Fullscreen or orientation lock failed:", err);
    }
  };

  const closeCanvasFullscreen = async () => {
    setIsMobileCanvasOpen(false);
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      }
      if (window.screen && window.screen.orientation && window.screen.orientation.unlock) {
        window.screen.orientation.unlock();
      }
    } catch (err) {
      console.warn("Exit fullscreen failed:", err);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 flex flex-col font-sans">
      {/* Top Navigation */}
      <Navbar
        connected={connected}
        status={status}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenDiagnostic={() => setIsDiagnosticOpen(true)}
        onResetMetrics={handleResetMetrics}
        isResetting={isResetting}
      />

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-5 space-y-4">
        {/* Connection and Hardware Status Panel */}
        <ConnectionPanel
          connected={connected}
          status={status}
          statusMessage={statusMessage}
          selectedPort={selectedPort}
          setSelectedPort={setSelectedPort}
          availablePorts={availablePorts}
          onRefreshPorts={fetchPorts}
          baudRate={baudRate}
          setBaudRate={setBaudRate}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
          framesReceived={framesReceived}
          errorsCount={errorsCount}
          isConnecting={isConnecting}
        />

        {/* Telemetry Metrics Row */}
        <TelemetryCards
          telemetry={telemetry}
          potholeCount={potholeCount}
          bumpCount={bumpCount}
          lastDepth={lastDepth}
        />

        {/* Mobile "Open Canvas" Button (Hidden on Desktop) */}
        <div className="block lg:hidden">
          <button
            onClick={openCanvasFullscreen}
            className="w-full flex items-center justify-center space-x-2 bg-zinc-900 hover:bg-zinc-800 text-white py-3 rounded-xl font-semibold shadow-sm transition"
          >
            <Maximize2 className="w-5 h-5" />
            <span>Open 2D Bike Canvas</span>
          </button>
        </div>

        {/* Desktop Embedded Canvas / Landscape Container */}
        <div className="hidden lg:block w-full bg-white rounded-xl shadow-sm border border-zinc-200 overflow-hidden">
          <RoadSimulation2D
            telemetry={telemetry}
            connected={connected}
            settings={settings}
            onSpeedChange={(speed) => handleSaveSettings({ ...settings, speed_kmph: speed })}
          />
        </div>

        {/* Telemetry Waveforms */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <h2 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
              Sensor Waveforms (10 Hz Telemetry)
            </h2>
            <span className="text-[10px] text-zinc-500 font-mono">
              Window: 100 samples
            </span>
          </div>
          <LiveCharts history={history} />
        </div>

        {/* Anomaly Detection Log Table */}
        <DetectionLog logs={logs} onClearLog={handleClearLog} />
      </main>

      {/* Mobile Fullscreen Canvas Overlay */}
      {isMobileCanvasOpen && (
        <div className="fixed inset-0 z-50 bg-black flex flex-col">
          <div className="flex items-center justify-between px-4 py-2 bg-zinc-900 text-white shadow-md z-10 shrink-0">
            <div className="flex flex-col">
              <span className="text-sm font-semibold">2D Bike Canvas</span>
            </div>
            <button
              onClick={closeCanvasFullscreen}
              className="p-1.5 rounded-full bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="flex-1 w-full bg-zinc-950 relative overflow-hidden flex items-center justify-center">
            {/* The RoadSimulation2D component should fit inside without scrolling */}
            <div className="w-full h-full max-h-[100vh] flex flex-col justify-center">
                <RoadSimulation2D
                  telemetry={telemetry}
                  connected={connected}
                  settings={settings}
                  onSpeedChange={(speed) => handleSaveSettings({ ...settings, speed_kmph: speed })}
                />
            </div>
          </div>
        </div>
      )}

      {/* Hardware Diagnostic Modal */}
      <DiagnosticModal
        isOpen={isDiagnosticOpen}
        onClose={() => setIsDiagnosticOpen(false)}
        selectedPort={selectedPort}
        baudRate={baudRate}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onSaveSettings={handleSaveSettings}
      />
    </div>
  );
}
