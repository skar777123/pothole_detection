import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import {
  Play,
  Pause,
  Plus,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Keyboard,
  X,
  Compass,
  Radio,
  ArrowUpDown,
  Zap,
  Clock,
  Gauge,
  ListFilter,
  Search,
  Download,
  Trash2,
  Ruler,
} from "lucide-react";

/**
 * RoadSimulation2D
 * ================
 * High-precision 2D road & vehicle simulation for LiDAR Pothole Detection.
 * 
 * Features:
 * - Fixed 5.0-meter (500 cm / 16.4 ft) LiDAR slant ray geometry.
 * - Prominent display of the sensor's baseline / base value (auto-calibrated or nominal).
 * - Full live Event Records & Anomaly Log table directly inside the simulation tab (synced with Dashboard).
 * - Clean, accurate Economic Commuter Motorcycle blueprint (100-125cc commuter architecture with perfectly aligned wheels, swingarm, dual shocks, engine, chaincase, fuel tank, long seat, forks, and TF02-Pro sensor).
 * - Multi-unit telemetry HUD (cm, m, ft, in, km/h, m/s, mph, ms).
 * - Right-to-left data registration (oncoming road ahead -> front wheel -> chassis -> rear wheel).
 */
export default function RoadSimulation2D({
  telemetry,
  connected,
  isSimulated,
  settings,
  resetTrigger,
  onResetSimulation,
  onSimulatedAnomaly,
  onSpeedChange,
  logs = [],
  onClearLog,
  potholeCount = 0,
  bumpCount = 0,
  lastDepth = 0,
}) {
  const canvasRef = useRef(null);

  // Hardware sync availability: only permitted when real physical sensor is detected
  const isHardwareAvailable = Boolean(connected && !isSimulated);

  // Simulation mode: "generator" (autonomous procedural road) or "hardware" (synced to live LiDAR)
  const [simMode, setSimMode] = useState("generator");
  const [isRunning, setIsRunning] = useState(true);
  const [simSpeedKmph, setSimSpeedKmph] = useState(30);
  const [autoSpawn, setAutoSpawn] = useState(false);
  const [showShortcutsModal, setShowShortcutsModal] = useState(false);

  // Filter state for the event records table
  const [eventFilter, setEventFilter] = useState("ALL");
  const [eventSearch, setEventSearch] = useState("");
  const roadPreset = "asphalt";
  // Automatically switch mode based on hardware availability
  useEffect(() => {
    if (isHardwareAvailable) {
      setSimMode("hardware");
    } else {
      setSimMode("generator");
    }
  }, [isHardwareAvailable]);

  // Sensor Base Value calculation (auto-calibrated baseline or 500 cm nominal slant)
  const baseValueCm = useMemo(() => {
    if (isHardwareAvailable && telemetry && typeof telemetry.baseline_cm === "number" && telemetry.baseline_cm > 0) {
      return telemetry.baseline_cm;
    }
    return 500.0; // Nominal 5.0m fixed beam baseline
  }, [isHardwareAvailable, telemetry]);

  const isCalibrated = Boolean(telemetry?.calibrated);
  const warmupCount = telemetry?.warmup_count || 0;
  const warmupTotal = telemetry?.warmup_total || 20;

  // Live Comprehensive HUD telemetry metrics (multi-unit)
  const [hudStats, setHudStats] = useState({
    speedKmph: 30,
    speedMps: 8.33,
    speedMph: 18.64,
    slantDistanceCm: 500.0,
    slantDistanceM: 5.0,
    slantDistanceFt: 16.4,
    baseValueCm: 500.0,
    baseValueM: 5.0,
    baseValueFt: 16.4,
    verticalDepthCm: 85.0,
    verticalDepthM: 0.85,
    verticalDepthIn: 33.46,
    deviationCm: 0.0,
    deviationMm: 0.0,
    deviationIn: 0.0,
    earlyWarningLeadCm: 492.7,
    earlyWarningLeadM: 4.93,
    earlyWarningLeadFt: 16.16,
    timeToImpactMs: 592,
    surfaceType: "Nominal Asphalt",
    isAlert: false,
    severity: "None",
    detectedCount: 0,
    activeHazard: null,
  });

  // Local session detections fallback buffer
  const [sessionDetections, setSessionDetections] = useState([]);

  // Combined event records list (merging logs prop and local session detections)
  const combinedEventRecords = useMemo(() => {
    const map = new Map();
    // Prioritize passed logs from App / Backend
    (logs || []).forEach((item) => {
      if (item && item.id) map.set(item.id, item);
    });
    // Add local session detections if not present
    sessionDetections.forEach((item) => {
      if (item && item.id && !map.has(item.id)) map.set(item.id, item);
    });
    const combined = Array.from(map.values());
    // Sort descending by id/timestamp
    return combined.sort((a, b) => (b.id || 0) - (a.id || 0));
  }, [logs, sessionDetections]);

  // Filtered event records
  const filteredEventRecords = useMemo(() => {
    return combinedEventRecords.filter((item) => {
      if (!item) return false;
      const t = String(item.type || "").toLowerCase();
      const timeStr = String(item.time || "").toLowerCase();
      const sev = String(item.severity || "").toLowerCase();

      const matchesFilter =
        eventFilter === "ALL" ||
        (eventFilter === "POTHOLE" && t.includes("pothole")) ||
        (eventFilter === "DEEP" && t.includes("deep")) ||
        (eventFilter === "BUMP" && t.includes("bump"));

      const q = eventSearch.trim().toLowerCase();
      const matchesSearch =
        !q ||
        t.includes(q) ||
        timeStr.includes(q) ||
        sev.includes(q) ||
        String(item.depth_cm || "").includes(q);

      return matchesFilter && matchesSearch;
    });
  }, [combinedEventRecords, eventFilter, eventSearch]);

  // Animation and physics state refs
  const animFrameRef = useRef(null);
  const lastTimeRef = useRef(performance.now());
  const distanceTraveledRef = useRef(0);
  const detectedCountRef = useRef(0);

  // Road terrain anomalies queue (worldX coordinates)
  const anomaliesRef = useRef([
    { id: 1, worldX: 950, type: "pothole", depthCm: 6.2, widthCm: 45, detected: false },
    { id: 2, worldX: 1800, type: "deep_pothole", depthCm: 12.0, widthCm: 65, detected: false },
    { id: 3, worldX: 2700, type: "bump", depthCm: -5.8, widthCm: 50, detected: false },
  ]);

  // Rolling terrain elevation buffer for hardware mode
  // Stores objects: { worldX, devCm } registered from RIGHT (laser hit point) to LEFT (wheels)
  const hardwareTerrainBufferRef = useRef([]);

  // Vehicle suspension dynamics
  const vehicleStateRef = useRef({
    pitch: 0,
    wheelRot: 0,
  });

  // Keep speed in sync with settings
  useEffect(() => {
    if (settings && settings.speed_kmph) {
      setSimSpeedKmph(settings.speed_kmph);
    }
  }, [settings?.speed_kmph]);

  // Push incoming live hardware telemetry into terrain buffer
  useEffect(() => {
    if (simMode === "hardware" && isHardwareAvailable && telemetry) {
      const dev = typeof telemetry.deviation_cm === "number" ? telemetry.deviation_cm : 0;
      const canvas = canvasRef.current;
      const width = canvas ? canvas.width / (window.devicePixelRatio || 1) : 1000;

      // Fixed 5-meter slant ray geometry
      const bikeScreenX = Math.max(90, width * 0.15);
      const lidarOriginX = bikeScreenX + 44;
      const nominalLeadDistanceCm = 492.7; // 500cm hypotenuse with ~85cm height
      const pixelsPerCm = 1.2; // 120 px = 1 meter
      const hitScreenX = lidarOriginX + nominalLeadDistanceCm * pixelsPerCm;

      const spawnWorldX = distanceTraveledRef.current + hitScreenX;

      hardwareTerrainBufferRef.current.push({
        worldX: spawnWorldX,
        devCm: dev,
      });

      // Keep buffer clean (last 400 points)
      if (hardwareTerrainBufferRef.current.length > 400) {
        hardwareTerrainBufferRef.current.shift();
      }
    }
  }, [simMode, isHardwareAvailable, telemetry]);

  // Manual anomaly spawner
  const spawnAnomaly = useCallback((type) => {
    const canvas = canvasRef.current;
    const viewWidth = canvas ? canvas.width / (window.devicePixelRatio || 1) : 1000;
    // Spawn ahead on the right side of the road
    const worldX = distanceTraveledRef.current + viewWidth + 150;

    let depthCm = 5.0;
    let widthCm = 45;

    if (type === "deep_pothole") {
      depthCm = 10.0 + Math.random() * 4.5;
      widthCm = 55 + Math.random() * 25;
    } else if (type === "pothole") {
      depthCm = 4.5 + Math.random() * 3.0;
      widthCm = 40 + Math.random() * 15;
    } else if (type === "bump") {
      depthCm = -(4.5 + Math.random() * 3.5);
      widthCm = 45 + Math.random() * 15;
    }

    anomaliesRef.current.push({
      id: Date.now() + Math.random(),
      worldX,
      type,
      depthCm: Math.round(depthCm * 10) / 10,
      widthCm: Math.round(widthCm),
      detected: false,
    });
  }, []);

  // Complete reset of simulation distance, anomaly queue, and session detections
  const handleReset = useCallback(() => {
    distanceTraveledRef.current = 0;
    detectedCountRef.current = 0;
    setSessionDetections([]);
    hardwareTerrainBufferRef.current = [];
    anomaliesRef.current = [
      { id: 1, worldX: 950, type: "pothole", depthCm: 6.2, widthCm: 45, detected: false },
      { id: 2, worldX: 1800, type: "deep_pothole", depthCm: 12.0, widthCm: 65, detected: false },
      { id: 3, worldX: 2700, type: "bump", depthCm: -5.8, widthCm: 50, detected: false },
    ];
    const nominalSurface =
      roadPreset === "mud"
        ? "Nominal Mud Track"
        : roadPreset === "dirt"
          ? "Nominal Dirt Road"
          : roadPreset === "cobble"
            ? "Nominal Cobblestone"
            : "Nominal Asphalt";

    setHudStats((prev) => ({
      ...prev,
      detectedCount: 0,
      deviationCm: 0.0,
      deviationMm: 0.0,
      deviationIn: 0.0,
      surfaceType: nominalSurface,
      isAlert: false,
      severity: "None",
      activeHazard: null,
    }));
  }, [roadPreset]);

  // Sync with global Reset button trigger
  useEffect(() => {
    if (resetTrigger && resetTrigger > 0) {
      handleReset();
    }
  }, [resetTrigger, handleReset]);

  // Export event records to CSV
  const exportCsv = () => {
    if (!combinedEventRecords || combinedEventRecords.length === 0) return;
    const headers = [
      "Time",
      "Type",
      "Deviation (cm)",
      "Depth (cm)",
      "Length (cm)",
      "Width (cm)",
      "Slant Range (cm)",
      "Lookahead Lead (m)",
      "Severity",
      "Confidence",
      "Signal Strength",
      "Baseline (cm)",
    ];
    const rows = combinedEventRecords.map((item) => [
      item.time || "",
      item.type || "",
      item.deviation_cm || "",
      item.depth_cm || 0,
      item.length_cm || 0,
      item.width_cm || 0,
      item.slant_range_cm || item.slant_range || "",
      item.lead_dist_m || "",
      item.severity || "",
      item.confidence || "",
      item.strength || 0,
      item.baseline || baseValueCm,
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `road_simulation_events_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Global Keyboard Shortcuts listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      const tag = e.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || e.target.isContentEditable) {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        setIsRunning((prev) => !prev);
        return;
      }

      if (e.key === "r" || e.key === "R") {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          handleReset();
          if (onResetSimulation) onResetSimulation();
          return;
        }
      }

      if (e.key === "1") {
        e.preventDefault();
        spawnAnomaly("pothole");
        return;
      }
      if (e.key === "2") {
        e.preventDefault();
        spawnAnomaly("deep_pothole");
        return;
      }
      if (e.key === "3") {
        e.preventDefault();
        spawnAnomaly("bump");
        return;
      }

      if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") {
        e.preventDefault();
        setSimSpeedKmph((curr) => {
          const next = Math.min(100, curr + 5);
          if (onSpeedChange) onSpeedChange(next);
          return next;
        });
        return;
      }
      if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") {
        e.preventDefault();
        setSimSpeedKmph((curr) => {
          const next = Math.max(5, curr - 5);
          if (onSpeedChange) onSpeedChange(next);
          return next;
        });
        return;
      }

      if (e.key === "h" || e.key === "H") {
        if (simMode === "generator") {
          e.preventDefault();
          setAutoSpawn((prev) => !prev);
          return;
        }
      }

      if (e.key === "p" || e.key === "P") {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          cycleRoadPreset();
          return;
        }
      }

      if (e.key === "?" || (e.shiftKey && e.key === "/")) {
        e.preventDefault();
        setShowShortcutsModal((prev) => !prev);
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [simMode, handleReset, onResetSimulation, spawnAnomaly, onSpeedChange]);

  // Main Canvas Rendering Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const render = (now) => {
      const dt = Math.min((now - lastTimeRef.current) / 1000, 0.1);
      lastTimeRef.current = now;

      // High-DPI scaling
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      // Real-world physical unit constants
      const speedMps = (simSpeedKmph * 1000) / 3600;
      const speedMph = simSpeedKmph * 0.621371;
      const pixelsPerMeter = 120; // 120 pixels on canvas = 1.0 meter
      const pixelsPerCm = pixelsPerMeter / 100; // 1.2 pixels = 1 cm

      if (isRunning) {
        const dx = speedMps * pixelsPerMeter * dt;
        distanceTraveledRef.current += dx;
        vehicleStateRef.current.wheelRot += (dx / 18) % (Math.PI * 2);

        // Procedural generator: spawn random potholes periodically
        if (simMode === "generator" && autoSpawn) {
          const spawnIntervalPx = 800;
          const lastAnomaly = anomaliesRef.current[anomaliesRef.current.length - 1];
          const nextSpawnThreshold = lastAnomaly
            ? lastAnomaly.worldX + spawnIntervalPx + Math.random() * 500
            : distanceTraveledRef.current + width + 250;

          if (distanceTraveledRef.current + width + 100 > nextSpawnThreshold) {
            const rand = Math.random();
            let newType = "pothole";
            if (rand < 0.45) newType = "pothole";
            else if (rand < 0.80) newType = "deep_pothole";
            else newType = "bump";
            spawnAnomaly(newType);
          }
        }
      }

      // Cleanup anomalies that scrolled off-screen to the left
      anomaliesRef.current = anomaliesRef.current.filter(
        (a) => a.worldX > distanceTraveledRef.current - 600
      );

      // Road geometry level on canvas
      const baselineY = height * 0.72;
      const nominalSensorHeightCm = 85.0; // Sensor height above road (0.85 m / 33.5 in)

      // FIXED 5.0 METER LIDAR RAY CONSTANTS
      const FIXED_SLANT_DIST_CM = 500.0; // 5.0 meters (500 cm / 16.4 ft)
      // Fixed inclination angle: sin(theta) = 85cm / 500cm = 0.170 -> theta ~ 9.79 deg
      const FIXED_BEAM_ANGLE_RAD = Math.asin(Math.min(0.95, nominalSensorHeightCm / FIXED_SLANT_DIST_CM));

      // Terrain elevation function: registers from RIGHT (oncoming ahead) to LEFT (bike)
      const getTerrainElevationAtScreenX = (screenX) => {
        const worldX = distanceTraveledRef.current + screenX;

        if (simMode === "hardware") {
          const buf = hardwareTerrainBufferRef.current;
          if (buf.length > 0) {
            const newestWorldX = buf[buf.length - 1].worldX;
            // Undiscovered road ahead of the laser scan is flat
            if (worldX > newestWorldX + 5) {
              return 0;
            }

            // Find closest buffered reading (queried from right to left)
            let matchDev = buf[buf.length - 1].devCm;
            for (let i = buf.length - 1; i >= 0; i--) {
              if (buf[i].worldX <= worldX) {
                matchDev = buf[i].devCm;
                break;
              }
            }
            return matchDev * pixelsPerCm;
          }
          return 0;
        }

        // Procedural baseline micro-texture based on road preset
        let baselineTexturePx = 0;
        if (roadPreset === "dirt") {
          baselineTexturePx = (Math.sin(worldX * 0.05) * 0.5 + Math.sin(worldX * 0.012) * 1.0) * pixelsPerCm;
        } else if (roadPreset === "mud") {
          baselineTexturePx = (Math.sin(worldX * 0.015) * 1.5 + Math.cos(worldX * 0.007) * 1.2) * pixelsPerCm;
        } else if (roadPreset === "cobble") {
          baselineTexturePx = (Math.sin(worldX * 0.10) * 0.4 + Math.cos(worldX * 0.02) * 0.6) * pixelsPerCm;
        }

        // Generator mode: realistic steep-walled crater / distinct speed bump profile
        let totalElevationPx = baselineTexturePx;
        for (const anom of anomaliesRef.current) {
          const distToCenter = worldX - anom.worldX;
          const halfWidthPx = (anom.widthCm * pixelsPerCm) / 2;

          if (Math.abs(distToCenter) <= halfWidthPx) {
            const normDist = Math.abs(distToCenter) / halfWidthPx; // 0 at center, 1.0 at outer rim
            let profileFactor = 0;

            if (anom.type === "pothole" || anom.type === "deep_pothole") {
              // REALISTIC STEEP-WALLED ASPHALT CRATER (Sharp edge drop, not a gentle slope)
              // Rim drop zone: outer 15% of crater width transitions steeply into cavity
              if (normDist > 0.85) {
                // Steep asphalt fracture wall drop (0 at rim edge to 0.92 at cavity wall)
                const wallT = (1.0 - normDist) / 0.15; // 0.0 to 1.0
                profileFactor = Math.pow(Math.sin(wallT * (Math.PI / 2)), 0.4);
              } else {
                // Interior crater cavity floor: flat/rough broken pavement with asphalt fracture noise
                const cavityNoise =
                  Math.sin(worldX * 0.4) * 0.06 + Math.cos(worldX * 0.85) * 0.04;
                profileFactor = 1.0 + cavityNoise;
              }
            } else if (anom.type === "bump") {
              // Defined trapezoidal / steep-edged speed hump
              if (normDist > 0.75) {
                const rampT = (1.0 - normDist) / 0.25;
                profileFactor = Math.sin(rampT * (Math.PI / 2));
              } else {
                profileFactor = 1.0 - Math.pow(normDist / 0.75, 2) * 0.12;
              }
            }

            const anomalyDepthPx = anom.depthCm * pixelsPerCm;
            totalElevationPx += anomalyDepthPx * profileFactor;
          }
        }
        return totalElevationPx;
      };

      // 1. DRAW CAD TECHNICAL BLUEPRINT BACKGROUND
      ctx.fillStyle = "#09090b";
      ctx.fillRect(0, 0, width, height);

      // Coordinate grid
      ctx.strokeStyle = "#18181b";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Top distance ruler in meters
      ctx.fillStyle = "#71717a";
      ctx.font = "9px monospace";
      const markerIntervalPx = 120;
      const startMarker = Math.floor(distanceTraveledRef.current / markerIntervalPx) * markerIntervalPx;
      for (let mx = startMarker; mx < distanceTraveledRef.current + width + markerIntervalPx; mx += markerIntervalPx) {
        const sx = mx - distanceTraveledRef.current;
        if (sx >= 0 && sx <= width) {
          ctx.beginPath();
          ctx.moveTo(sx, 16);
          ctx.lineTo(sx, 24);
          ctx.strokeStyle = "#3f3f46";
          ctx.stroke();
          ctx.fillText(`${(mx / pixelsPerMeter).toFixed(1)}m`, sx + 3, 24);
        }
      }

      // 2. DRAW ROAD TERRAIN CROSS-SECTION (High-resolution step for crisp crater edges)
      const step = 2; // 2px fine sampling ensures sharp vertical crater walls
      const surfacePoints = [];
      for (let sx = 0; sx <= width + step; sx += step) {
        const elev = getTerrainElevationAtScreenX(sx);
        surfacePoints.push({ x: sx, y: baselineY + elev });
      }

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(0, height);
      ctx.lineTo(0, surfacePoints[0].y);
      for (const p of surfacePoints) {
        ctx.lineTo(p.x, p.y);
      }
      ctx.lineTo(width, height);
      ctx.closePath();
      ctx.clip();

      // Road sub-base hatching
      ctx.fillStyle = "#121215";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#27272a";
      ctx.lineWidth = 1;
      for (let x = -height; x < width + height; x += 24) {
        ctx.beginPath();
        ctx.moveTo(x, baselineY - 20);
        ctx.lineTo(x + height, height);
        ctx.stroke();
      }
      ctx.restore();

      // Primary road surface boundary line
      ctx.strokeStyle =
        roadPreset === "mud"
          ? "#d97706"
          : roadPreset === "dirt"
            ? "#b45309"
            : roadPreset === "cobble"
              ? "#a1a1aa"
              : "#e4e4e7";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(surfacePoints[0].x, surfacePoints[0].y);
      for (let i = 1; i < surfacePoints.length; i++) {
        ctx.lineTo(surfacePoints[i].x, surfacePoints[i].y);
      }
      ctx.stroke();

      // Highlight crater cavity fracture markers and sharp rims
      for (const anom of anomaliesRef.current) {
        const screenCenterX = anom.worldX - distanceTraveledRef.current;
        const halfWidthPx = (anom.widthCm * pixelsPerCm) / 2;
        const leftRimX = screenCenterX - halfWidthPx;
        const rightRimX = screenCenterX + halfWidthPx;

        if (rightRimX >= 0 && leftRimX <= width) {
          const isPotholeHazard = anom.type.includes("pothole");
          const isDeepHazard = anom.type === "deep_pothole";

          if (isPotholeHazard) {
            // Draw red/amber sharp rim boundary ticks
            ctx.strokeStyle = isDeepHazard ? "#ef4444" : "#f59e0b";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(leftRimX, baselineY - 4);
            ctx.lineTo(leftRimX, baselineY + 4);
            ctx.moveTo(rightRimX, baselineY - 4);
            ctx.lineTo(rightRimX, baselineY + 4);
            ctx.stroke();

            // Cavity depth marker line
            const cavityBottomY = baselineY + anom.depthCm * pixelsPerCm;
            ctx.strokeStyle = isDeepHazard ? "rgba(239, 68, 68, 0.4)" : "rgba(245, 158, 11, 0.4)";
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(leftRimX + 4, cavityBottomY);
            ctx.lineTo(rightRimX - 4, cavityBottomY);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // Flat road nominal baseline reference (dashed yellow) with Base Value label
      ctx.strokeStyle = "rgba(234, 179, 8, 0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(0, baselineY);
      ctx.lineTo(width, baselineY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(234, 179, 8, 0.7)";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`Nominal Ground Baseline (d₀ = ${baseValueCm.toFixed(1)} cm)`, 12, baselineY - 6);

      // 3. MOTORBIKE POSITION & SUSPENSION DYNAMICS
      // Economic Commuter Motorcycle geometry (Wheelbase: 110px, Wheel radius: 18px)
      const bikeScreenX = Math.max(90, width * 0.15);
      const wheelRadius = 18;
      const rearAxleX = bikeScreenX - 55;
      const frontAxleX = bikeScreenX + 55;

      const rearGroundY = baselineY + getTerrainElevationAtScreenX(rearAxleX);
      const frontGroundY = baselineY + getTerrainElevationAtScreenX(frontAxleX);

      const rearAxleY = rearGroundY - wheelRadius;
      const frontAxleY = frontGroundY - wheelRadius;

      // Chassis pitch angle from road slope
      const targetPitch = Math.atan2(frontGroundY - rearGroundY, frontAxleX - rearAxleX);
      vehicleStateRef.current.pitch += (targetPitch - vehicleStateRef.current.pitch) * 0.2;
      const pitch = vehicleStateRef.current.pitch;

      const chassisMidX = (rearAxleX + frontAxleX) / 2;
      const chassisMidY = (rearAxleY + frontAxleY) / 2;

      // Draw Detailed Commuter Wheel (5-Spoke Star Alloy + Radial Treads + Caliper + Center Hub)
      const drawCommuterWheel = (wx, wy) => {
        ctx.save();
        ctx.translate(wx, wy);
        ctx.rotate(vehicleStateRef.current.wheelRot);

        // Outer Rubber Tire
        ctx.beginPath();
        ctx.arc(0, 0, wheelRadius, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.2;
        ctx.stroke();

        // Inner Alloy Rim Circle
        ctx.beginPath();
        ctx.arc(0, 0, wheelRadius - 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#a1a1aa";
        ctx.lineWidth = 1.2;
        ctx.stroke();

        // 16 Tire Tread Grooves along circumference
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.2;
        const numTreads = 16;
        for (let i = 0; i < numTreads; i++) {
          const a = (i * 2 * Math.PI) / numTreads;
          const r1 = wheelRadius - 2;
          const r2 = wheelRadius + 1.6;
          ctx.beginPath();
          ctx.moveTo(Math.cos(a) * r1, Math.sin(a) * r1);
          ctx.lineTo(Math.cos(a) * r2, Math.sin(a) * r2);
          ctx.stroke();
        }

        // 5-Spoke Commuter Alloy Pattern
        ctx.strokeStyle = "#e4e4e7";
        ctx.lineWidth = 1.6;
        const numSpokes = 5;
        for (let i = 0; i < numSpokes; i++) {
          const a = (i * 2 * Math.PI) / numSpokes;
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(Math.cos(a) * (wheelRadius - 4.5), Math.sin(a) * (wheelRadius - 4.5));
          ctx.stroke();
        }

        // Disc Rotor / Drum Rim Ring
        ctx.beginPath();
        ctx.arc(0, 0, wheelRadius - 8.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#52525b";
        ctx.lineWidth = 1;
        ctx.stroke();

        // Center Axle Hub Nut
        ctx.beginPath();
        ctx.arc(0, 0, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();

        ctx.restore();
      };

      drawCommuterWheel(rearAxleX, rearAxleY);
      drawCommuterWheel(frontAxleX, frontAxleY);

      // 4. DRAW ECONOMIC COMMUTER MOTORCYCLE BLUEPRINT (100-125cc commuter architecture)
      // All local coordinates relative to (chassisMidX, chassisMidY).
      // Rear Axle is strictly at (-55, 0) and Front Axle is strictly at (55, 0).
      ctx.save();
      ctx.translate(chassisMidX, chassisMidY);
      ctx.rotate(pitch);

      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";

      // A. REAR WHEEL MUDGUARD (Concentric to rear axle at -55, 0)
      ctx.beginPath();
      ctx.arc(-55, 0, wheelRadius + 3.5, -Math.PI * 0.90, -Math.PI * 0.20);
      ctx.stroke();

      // Rear Red Commuter Tail Lamp & Reflector
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(-60, -16, 5, 8);
      ctx.strokeStyle = "#b91c1c";
      ctx.strokeRect(-60, -16, 5, 8);

      // License Plate / Tidy Bracket
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(-58, -8);
      ctx.lineTo(-64, -2);
      ctx.stroke();

      // B. REAR COMMUTER SWINGARM (From frame pivot at -10, 4 directly to rear axle at -55, 0)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(-10, 4);
      ctx.lineTo(-55, 0);
      ctx.stroke();

      // C. COMMUTER ENCLOSED CHAINCASE (Metal protective box from engine to rear axle)
      ctx.strokeStyle = "#a1a1aa";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(-6, 4);
      ctx.lineTo(-55, -2);
      ctx.lineTo(-55, 2);
      ctx.lineTo(-6, 7);
      ctx.closePath();
      ctx.stroke();

      // D. DUAL REAR SHOCK ABSORBERS (From subframe mount at -36, -14 straight to swingarm at -50, 0)
      // Chrome Damper Rod
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(-36, -14);
      ctx.lineTo(-50, 0);
      ctx.stroke();

      // Helical Coil Spring Windings
      ctx.strokeStyle = "#d4d4d8";
      ctx.lineWidth = 1.8;
      const shockSteps = 5;
      for (let i = 1; i <= shockSteps; i++) {
        const t = i / (shockSteps + 1);
        const sx = -36 + t * (-50 - -36);
        const sy = -14 + t * (0 - -14);
        ctx.beginPath();
        ctx.moveTo(sx - 2, sy - 2);
        ctx.lineTo(sx + 2, sy + 2);
        ctx.stroke();
      }

      // E. ENGINE & TRANSMISSION BLOCK (125cc Horizontal-Slanted Commuter Engine)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      // Crankcase
      ctx.beginPath();
      ctx.rect(-12, 2, 20, 10);
      ctx.stroke();

      // Horizontal Cylinder & Head with Air Cooling Fins
      ctx.beginPath();
      ctx.moveTo(8, 2);
      ctx.lineTo(20, -2);
      ctx.lineTo(20, 6);
      ctx.lineTo(8, 10);
      ctx.closePath();
      ctx.stroke();

      // Cylinder Cooling Fins
      ctx.strokeStyle = "#a1a1aa";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(10, 1); ctx.lineTo(10, 9);
      ctx.moveTo(13, 0); ctx.lineTo(13, 8);
      ctx.moveTo(16, -1); ctx.lineTo(16, 7);
      ctx.stroke();

      // Carburetor & Air Intake
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.rect(0, -6, 6, 6);
      ctx.stroke();

      // Rider Footpeg
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(-2, 10);
      ctx.lineTo(-2, 16);
      ctx.lineTo(4, 16);
      ctx.stroke();

      // F. LONG COMMUTER EXHAUST SYSTEM & CHROME HEAT SHIELD
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(18, 2); // Cylinder exhaust port
      ctx.lineTo(12, 12); // Header downpipe curve
      ctx.lineTo(-4, 12); // Underbelly pipe
      ctx.lineTo(-56, 4); // Long straight commuter muffler
      ctx.stroke();

      // Chrome Slotted Heat Guard Plate
      ctx.strokeStyle = "#a1a1aa";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(-16, 10);
      ctx.lineTo(-46, 6);
      ctx.stroke();

      // G. COMMUTER TUBULAR FRAME (Backbone & Twin Cradle)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(30, -24); // Steering headstock
      ctx.lineTo(8, -14);  // Backbone tube
      ctx.lineTo(-10, 4);  // Swingarm pivot bracket
      ctx.moveTo(30, -24);
      ctx.lineTo(12, 8);   // Down-tube cradle
      ctx.lineTo(-12, 8);  // Engine cradle
      ctx.stroke();

      // H. COMMUTER FUEL TANK (Sleek utilitarian profile)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(30, -24); // Headstock junction
      ctx.lineTo(14, -28); // Tank top peak
      ctx.lineTo(-6, -16); // Seat nose junction
      ctx.lineTo(4, -14);  // Knee pad lower edge
      ctx.lineTo(24, -14);
      ctx.closePath();
      ctx.stroke();

      // Chrome Fuel Cap
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(16, -30, 2.5, 0, Math.PI * 2);
      ctx.stroke();

      // Tank Rubber Knee Grip Badge
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(10, -22);
      ctx.lineTo(20, -20);
      ctx.lineTo(18, -16);
      ctx.lineTo(8, -16);
      ctx.closePath();
      ctx.stroke();

      // I. LONG FLAT COMMUTER SEAT (Rider + Pillion with Rear Carrier)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(-6, -16);  // Front nose
      ctx.lineTo(-46, -16); // Seat top profile (flat & comfortable)
      ctx.lineTo(-46, -11); // Seat base tail
      ctx.lineTo(-6, -11);  // Seat base front
      ctx.closePath();
      ctx.stroke();

      // Side Battery Utility Panel below seat
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.rect(-24, -10, 16, 12);
      ctx.stroke();

      // Chrome Luggage Carrier / Pillion Grab Rail
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(-38, -16);
      ctx.lineTo(-58, -18);
      ctx.lineTo(-58, -14);
      ctx.lineTo(-46, -12);
      ctx.stroke();

      // J. FRONT TELESCOPIC FORK ASSEMBLY (From headstock 30,-24 straight to front axle 55,0)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(30, -24); // Triple clamp
      ctx.lineTo(55, 0);   // Front axle hub
      ctx.stroke();

      // Rubber Accordion Fork Gaiter Boots
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(38, -16);
      ctx.lineTo(44, -10);
      ctx.stroke();

      // K. FRONT WHEEL MUDGUARD (Concentric to front axle at 55, 0)
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(55, 0, wheelRadius + 3.5, -Math.PI * 0.80, -Math.PI * 0.15);
      ctx.stroke();

      // Mudguard Mounting Stay Strut
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(55, 0);
      ctx.lineTo(44, -14);
      ctx.stroke();

      // L. HANDLEBARS, CONTROLS & TWIN GAUGES
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(30, -24); // Clamp
      ctx.lineTo(26, -36); // Handlebar riser
      ctx.lineTo(20, -38); // Grip
      ctx.stroke();

      // Brake Lever
      ctx.strokeStyle = "#a1a1aa";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(24, -37);
      ctx.lineTo(28, -39);
      ctx.stroke();

      // Commuter Twin Round Instrument Dials (Speedometer / Fuel Gauge)
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(28, -38, 2.5, 0, Math.PI * 2);
      ctx.stroke();

      // Round Chrome Rear-View Mirror
      ctx.strokeStyle = "#e4e4e7";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(26, -36);
      ctx.lineTo(20, -48);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(18, -50, 3, 0, Math.PI * 2);
      ctx.stroke();

      // M. COMMUTER HEADLIGHT & AMBER INDICATORS
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(32, -24);
      ctx.lineTo(42, -20);
      ctx.lineTo(42, -14);
      ctx.lineTo(34, -16);
      ctx.closePath();
      ctx.stroke();

      // Yellow Headlight Glass Lens
      ctx.strokeStyle = "#fef08a";
      ctx.lineWidth = 2.0;
      ctx.beginPath();
      ctx.moveTo(42, -20);
      ctx.lineTo(42, -14);
      ctx.stroke();

      // Amber Turn Signal Indicator
      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(36, -26, 2, 0, Math.PI * 2);
      ctx.fill();

      // N. TF02-PRO LIDAR SENSOR UNIT & RIGID MOUNT (Mounted above headlight shelf)
      const sensorLocalX = 44;
      const sensorLocalY = -30;

      ctx.save();
      ctx.translate(sensorLocalX, sensorLocalY);
      ctx.rotate(FIXED_BEAM_ANGLE_RAD);

      // LiDAR Mounting Shelf Bracket
      ctx.strokeStyle = "#71717a";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(-6, 4);
      ctx.lineTo(0, 0);
      ctx.stroke();

      // TF02-Pro Housing
      ctx.fillStyle = "#09090b";
      ctx.fillRect(-4, -4, 12, 8);
      ctx.strokeStyle = "#f97316"; // Fiery red-amber LiDAR casing accent
      ctx.lineWidth = 1.4;
      ctx.strokeRect(-4, -4, 12, 8);

      // Red Laser Emitter Lens (Facing right-downward)
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(8, 0, 2.2, 0, Math.PI * 2);
      ctx.fill();

      // Amber Glow Halo
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(8, 0, 3.2, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
      ctx.restore(); // Exit motorbike transform

      // 5. FIXED 5.0 METER LIDAR LASER BEAM RAYCAST
      const cosP = Math.cos(pitch);
      const sinP = Math.sin(pitch);
      const sensorLocalOffsetX = 44;
      const sensorLocalOffsetY = -30;

      const lidarOriginX = chassisMidX + sensorLocalOffsetX * cosP - sensorLocalOffsetY * sinP;
      const lidarOriginY = chassisMidY + sensorLocalOffsetX * sinP + sensorLocalOffsetY * cosP;

      const effectiveBeamAngleRad = FIXED_BEAM_ANGLE_RAD + pitch;
      const cosBeam = Math.cos(effectiveBeamAngleRad);
      const sinBeam = Math.sin(effectiveBeamAngleRad);

      // Ray intersection with oncoming road terrain ahead to the right
      let rayT = (baselineY - lidarOriginY) / Math.max(sinBeam, 0.05);
      let hitX = lidarOriginX + rayT * cosBeam;
      let hitY = baselineY + getTerrainElevationAtScreenX(hitX);

      // Multi-pass crater boundary refinement
      for (let iter = 0; iter < 4; iter++) {
        const terrainAtX = baselineY + getTerrainElevationAtScreenX(hitX);
        const errorY = terrainAtX - (lidarOriginY + rayT * sinBeam);
        rayT += errorY / Math.max(sinBeam, 0.05);
        hitX = lidarOriginX + rayT * cosBeam;
        hitY = baselineY + getTerrainElevationAtScreenX(hitX);
      }

      // Real physical distance calculations
      const measuredSlantPx = Math.hypot(hitX - lidarOriginX, hitY - lidarOriginY);
      const measuredSlantCm = measuredSlantPx / pixelsPerCm;
      const measuredSlantM = measuredSlantCm / 100;
      const measuredSlantFt = measuredSlantCm / 30.48;

      const verticalHeightCm = (hitY - lidarOriginY) / pixelsPerCm;
      const verticalHeightM = verticalHeightCm / 100;
      const verticalHeightIn = verticalHeightCm / 2.54;

      const deltaVerticalCm = (hitY - baselineY) / pixelsPerCm;
      const deltaVerticalMm = deltaVerticalCm * 10;
      const deltaVerticalIn = deltaVerticalCm / 2.54;

      const leadDistanceCm = Math.max(0, (hitX - frontAxleX) / pixelsPerCm);
      const leadDistanceM = leadDistanceCm / 100;
      const leadDistanceFt = leadDistanceCm / 30.48;

      const timeToImpactMs = speedMps > 0 ? Math.round((leadDistanceM / speedMps) * 1000) : 9999;

      // Anomaly thresholds
      const potThresh = settings?.pot_thresh || 4.5;
      const deepThresh = settings?.deep_thresh || 8.0;
      const bumpThresh = settings?.bump_thresh || 4.5;

      const isPothole = deltaVerticalCm > potThresh;
      const isDeep = deltaVerticalCm > deepThresh;
      const isBump = deltaVerticalCm < -bumpThresh;
      const isAlert = isPothole || isBump;

      // Surface Classification
      let nominalTitle = "Nominal Asphalt";
      if (roadPreset === "mud") nominalTitle = "Nominal Mud Track";
      else if (roadPreset === "dirt") nominalTitle = "Nominal Dirt Road";
      else if (roadPreset === "cobble") nominalTitle = "Nominal Cobblestone";

      let activeClass = nominalTitle;
      let severityLabel = "None";
      if (isDeep) {
        severityLabel = "Critical";
        activeClass = roadPreset === "mud" ? "Deep Mud Rut" : roadPreset === "dirt" ? "Severe Washout" : roadPreset === "cobble" ? "Sunken Paver Pit" : "Deep Pothole";
      } else if (isPothole) {
        severityLabel = "Moderate";
        activeClass = roadPreset === "mud" ? "Mud Pothole" : roadPreset === "dirt" ? "Gravel Depression" : roadPreset === "cobble" ? "Loose Paver Hole" : "Shallow Pothole";
      } else if (isBump) {
        severityLabel = "Caution";
        activeClass = roadPreset === "mud" ? "Mud Ridge" : roadPreset === "dirt" ? "Gravel Mound" : roadPreset === "cobble" ? "Raised Paver Stone" : "Speed Bump";
      }

      // Generator mode anomaly confirmation
      let activeScannedHazard = null;
      if (simMode === "generator") {
        for (const anom of anomaliesRef.current) {
          const worldHitX = distanceTraveledRef.current + hitX;
          const halfWidthPx = (anom.widthCm * pixelsPerCm) / 2;
          if (Math.abs(worldHitX - anom.worldX) < halfWidthPx) {
            activeScannedHazard = anom;
            if (!anom.detected) {
              anom.detected = true;
              detectedCountRef.current += 1;

              const isDeepAnom = anom.type === "deep_pothole";
              const isPotholeAnom = anom.type === "pothole";
              let detectedTypeName = "Speed Bump";
              if (isDeepAnom) {
                detectedTypeName = roadPreset === "mud" ? "Deep Mud Rut" : roadPreset === "dirt" ? "Severe Washout" : roadPreset === "cobble" ? "Sunken Paver Pit" : "Deep Pothole";
              } else if (isPotholeAnom) {
                detectedTypeName = roadPreset === "mud" ? "Mud Pothole" : roadPreset === "dirt" ? "Gravel Depression" : roadPreset === "cobble" ? "Loose Paver Hole" : "Shallow Pothole";
              } else {
                detectedTypeName = roadPreset === "mud" ? "Mud Ridge" : roadPreset === "dirt" ? "Gravel Mound" : roadPreset === "cobble" ? "Raised Paver Stone" : "Speed Bump";
              }

              const detectedObj = {
                id: Date.now(),
                time: new Date().toLocaleTimeString(),
                type: detectedTypeName,
                deviation_cm: `${anom.depthCm > 0 ? "+" : ""}${anom.depthCm.toFixed(1)}`,
                depth_cm: Math.abs(anom.depthCm),
                depth_in: Math.round((Math.abs(anom.depthCm) / 2.54) * 10) / 10,
                length_cm: anom.widthCm,
                length_ft: Math.round((anom.widthCm / 30.48) * 10) / 10,
                width_cm: Math.round(anom.widthCm * 0.8),
                severity: Math.abs(anom.depthCm) >= 8.0 ? "Deep / Dangerous" : "Shallow",
                confidence: "98%",
                strength: 1100,
                baseline: `${baseValueCm.toFixed(0)}`,
                slant_range_cm: Math.round(measuredSlantCm * 10) / 10,
                slant_range_m: Math.round(measuredSlantM * 100) / 100,
                slant_range_ft: Math.round(measuredSlantFt * 10) / 10,
                lead_dist_m: Math.round(leadDistanceM * 100) / 100,
                lead_dist_ft: Math.round(leadDistanceFt * 10) / 10,
              };

              setSessionDetections((prev) => [detectedObj, ...prev.slice(0, 49)]);

              if (onSimulatedAnomaly) {
                onSimulatedAnomaly(detectedObj);
              }
            }
          }
        }
      }

      // Update state for HUD display
      setHudStats({
        speedKmph: simSpeedKmph,
        speedMps: Math.round(speedMps * 100) / 100,
        speedMph: Math.round(speedMph * 10) / 10,
        slantDistanceCm: Math.round(measuredSlantCm * 10) / 10,
        slantDistanceM: Math.round(measuredSlantM * 100) / 100,
        slantDistanceFt: Math.round(measuredSlantFt * 10) / 10,
        baseValueCm: Math.round(baseValueCm * 10) / 10,
        baseValueM: Math.round((baseValueCm / 100) * 100) / 100,
        baseValueFt: Math.round((baseValueCm / 30.48) * 10) / 10,
        verticalDepthCm: Math.round(verticalHeightCm * 10) / 10,
        verticalDepthM: Math.round(verticalHeightM * 100) / 100,
        verticalDepthIn: Math.round(verticalHeightIn * 10) / 10,
        deviationCm: Math.round(deltaVerticalCm * 10) / 10,
        deviationMm: Math.round(deltaVerticalMm),
        deviationIn: Math.round(deltaVerticalIn * 100) / 100,
        earlyWarningLeadCm: Math.round(leadDistanceCm),
        earlyWarningLeadM: Math.round(leadDistanceM * 100) / 100,
        earlyWarningLeadFt: Math.round(leadDistanceFt * 10) / 10,
        timeToImpactMs,
        surfaceType: activeClass,
        isAlert,
        severity: severityLabel,
        detectedCount: simMode === "generator" ? detectedCountRef.current : (potholeCount + bumpCount),
        activeHazard: activeScannedHazard,
      });

      // 6. DRAW LASER BEAM (FIERY RED-AMBER MIXED)
      ctx.save();
      // Outer glow
      ctx.strokeStyle = isDeep ? "rgba(239, 68, 68, 0.45)" : isAlert ? "rgba(245, 158, 11, 0.45)" : "rgba(249, 115, 22, 0.35)";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(lidarOriginX, lidarOriginY);
      ctx.lineTo(hitX, hitY);
      ctx.stroke();

      // Inner intense core beam
      ctx.strokeStyle = isDeep ? "#ef4444" : isAlert ? "#f59e0b" : "#ffedd5";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(lidarOriginX, lidarOriginY);
      ctx.lineTo(hitX, hitY);
      ctx.stroke();

      // Laser impact spot on ground
      ctx.fillStyle = isDeep ? "#ef4444" : isAlert ? "#f59e0b" : "#f97316";
      ctx.beginPath();
      ctx.arc(hitX, hitY, 4, 0, Math.PI * 2);
      ctx.fill();

      // Impact halo rings
      ctx.strokeStyle = isDeep ? "rgba(239, 68, 68, 0.6)" : "rgba(249, 115, 22, 0.6)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hitX, hitY, 7, 0, Math.PI * 2);
      ctx.stroke();

      // Mid-beam telemetry label
      const midX = (lidarOriginX + hitX) / 2;
      const midY = (lidarOriginY + hitY) / 2;
      ctx.fillStyle = "#f97316";
      ctx.font = "bold 9px monospace";
      ctx.fillText(`5.0m Ray (R: ${measuredSlantM.toFixed(2)}m / ${measuredSlantFt.toFixed(1)}ft)`, midX + 12, midY - 6);

      // 7. FLOATING HAZARD CALLOUT OVER ACTIVE DETECTED ROAD ANOMALY
      if (isAlert) {
        ctx.save();
        ctx.translate(hitX, hitY - 32);

        // Callout Tag Background
        ctx.fillStyle = isDeep ? "rgba(220, 38, 38, 0.95)" : "rgba(217, 119, 6, 0.95)";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(-55, -14, 110, 24, [4]);
        ctx.fill();
        ctx.stroke();

        // Pin pointer triangle
        ctx.beginPath();
        ctx.moveTo(-5, 10);
        ctx.lineTo(0, 18);
        ctx.lineTo(5, 10);
        ctx.fill();

        // Text in Callout
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(
          `${deltaVerticalCm > 0 ? "POTHOLE" : "BUMP"} ${Math.abs(deltaVerticalCm).toFixed(1)}cm`,
          0,
          -2
        );
        ctx.font = "8px monospace";
        ctx.fillText(`Lead: ${leadDistanceM.toFixed(2)}m | ${timeToImpactMs}ms`, 0, 7);

        ctx.restore();
      }

      ctx.restore();

      ctx.restore(); // Restore high-DPI scaling
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [
    isRunning,
    simSpeedKmph,
    simMode,
    autoSpawn,
    settings,
    spawnAnomaly,
    onSimulatedAnomaly,
    isHardwareAvailable,
    roadPreset,
    baseValueCm,
    potholeCount,
    bumpCount,
  ]);

  return (
    <div className="space-y-4 font-sans">
      {/* Simulation Master Control Bar */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3 shadow-xs">
        {/* Left: Play / Reset / Presets */}
        <div className="flex items-center space-x-2 flex-wrap gap-y-2">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition ${isRunning
              ? "bg-zinc-900 text-white hover:bg-zinc-800 shadow-xs"
              : "bg-emerald-600 text-white hover:bg-emerald-500 shadow-xs"
              }`}
          >
            {isRunning ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>{isRunning ? "Pause (Space)" : "Resume (Space)"}</span>
          </button>

          <button
            onClick={handleReset}
            title="Reset Simulation Distance and Anomaly Log (Key R)"
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-zinc-50 border border-zinc-300 transition"
          >
            <RefreshCw className="w-3.5 h-3.5 text-zinc-500" />
            <span className="hidden sm:inline">Reset</span>
          </button>
        </div>
      </div>

      {/* Primary LiDAR Telemetry & Base Value Metric Bar (7 Cards) */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 shadow-xs font-mono">
        {/* 1. Sensor Baseline / Base Value (d0) */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Sensor Base (d₀)</span>
            <Compass className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-amber-700 block mt-0.5">{hudStats.baseValueCm} cm</span>
          <span className="text-[10px] text-zinc-500 truncate block">
            {isHardwareAvailable
              ? isCalibrated
                ? "Auto-Calibrated (20-pt mean)"
                : `Calibrating (${warmupCount}/${warmupTotal})`
              : `${hudStats.baseValueM} m | ${hudStats.baseValueFt} ft`}
          </span>
        </div>

        {/* 2. Measured Slant Distance (R) */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Measured Slant (R)</span>
            <Radio className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-orange-600 block mt-0.5">{hudStats.slantDistanceCm} cm</span>
          <span className="text-[10px] text-zinc-500 block">{hudStats.slantDistanceM} m | {hudStats.slantDistanceFt} ft</span>
        </div>

        {/* 3. Surface Deviation (Delta d) */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Surface Dev (Δd)</span>
            <ArrowUpDown className="w-3 h-3 text-zinc-400" />
          </div>
          <span className={`text-xs font-bold block mt-0.5 ${hudStats.deviationCm > 4.5 ? "text-rose-600" : hudStats.deviationCm < -4.5 ? "text-blue-600" : "text-zinc-900"}`}>
            {hudStats.deviationCm > 0 ? `+${hudStats.deviationCm}` : hudStats.deviationCm} cm
          </span>
          <span className="text-[10px] text-zinc-500 block">{hudStats.deviationMm} mm | {hudStats.deviationIn} in</span>
        </div>

        {/* 4. Lookahead Distance */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Lookahead Lead</span>
            <Ruler className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-emerald-700 block mt-0.5">{hudStats.earlyWarningLeadM} m</span>
          <span className="text-[10px] text-zinc-500 block">{hudStats.earlyWarningLeadCm} cm | {hudStats.earlyWarningLeadFt} ft</span>
        </div>

        {/* 5. Warning Reaction Window */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Time to Impact</span>
            <Clock className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-indigo-700 block mt-0.5">{hudStats.timeToImpactMs} ms</span>
          <span className="text-[10px] text-zinc-500 block">{(hudStats.timeToImpactMs / 1000).toFixed(2)} s reaction</span>
        </div>

        {/* 6. Current Speed */}
        <div className="border-r border-zinc-100 pr-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Vehicle Speed</span>
            <Gauge className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-zinc-900 block mt-0.5">{hudStats.speedKmph} km/h</span>
          <span className="text-[10px] text-zinc-500 block">{hudStats.speedMps} m/s | {hudStats.speedMph} mph</span>
        </div>

        {/* 7. Total Detected Anomalies */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-zinc-400 uppercase tracking-wider block">Event Counts</span>
            <Zap className="w-3 h-3 text-zinc-400" />
          </div>
          <span className="text-xs font-bold text-rose-600 block mt-0.5">{hudStats.detectedCount}</span>
          <span className="text-[10px] text-zinc-500 block">
            {potholeCount} holes | {bumpCount} bumps
          </span>
        </div>
      </div>

      {/* Main 2D Canvas Viewport */}
      <div className="relative w-full h-[430px] bg-black border border-zinc-800 rounded-lg overflow-hidden shadow-sm">
        <canvas ref={canvasRef} className="w-full h-full block cursor-crosshair" />

        {/* Top Left Live HUD Badges */}
        <div className="absolute top-3 left-3 flex items-center space-x-2 pointer-events-none">
          <div
            className={`px-3 py-1.5 rounded-md border text-xs font-mono font-semibold flex items-center space-x-2 shadow-md ${hudStats.isAlert
              ? hudStats.surfaceType.includes("Deep")
                ? "bg-red-600 text-white border-red-700 animate-pulse"
                : "bg-amber-500 text-white border-amber-600"
              : "bg-zinc-900/95 text-zinc-100 border-zinc-700"
              }`}
          >
            {hudStats.isAlert ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
            <span>{hudStats.surfaceType}</span>
          </div>

          <div className="bg-zinc-900/90 text-zinc-200 border border-zinc-700 px-3 py-1.5 rounded-md text-xs font-mono shadow-md hidden sm:block">
            Base d₀: <span className="font-bold text-amber-400">{hudStats.baseValueCm} cm</span>
            <span className="mx-2 text-zinc-600">|</span>
            Slant R: <span className="font-bold text-white">{hudStats.slantDistanceCm} cm</span>
            <span className="mx-2 text-zinc-600">|</span>
            Lead: <span className="font-bold text-emerald-400">{hudStats.earlyWarningLeadM} m</span>
          </div>
        </div>

        {/* Top Right Early Warning and Session Counts */}
        <div className="absolute top-3 right-3 flex items-center space-x-2 pointer-events-none">
          <div className="bg-zinc-900/90 text-zinc-200 border border-zinc-700 px-2.5 py-1.5 rounded-md text-xs font-mono shadow-md">
            Speed: <span className="font-semibold text-white">{hudStats.speedKmph} km/h</span>
          </div>
          <div className="bg-zinc-900/90 text-zinc-200 border border-zinc-700 px-2.5 py-1.5 rounded-md text-xs font-mono shadow-md">
            Events: <span className="font-semibold text-rose-400">{hudStats.detectedCount}</span>
          </div>
        </div>

        {/* Bottom Left Speed Slider */}
        <div className="absolute bottom-3 left-3 bg-zinc-900/95 border border-zinc-700 px-3 py-1.5 rounded-md flex items-center space-x-2.5 text-xs font-mono text-zinc-300 shadow-md">
          <span className="text-zinc-400">Speed (W/S):</span>
          <input
            type="range"
            min="10"
            max="100"
            step="5"
            value={simSpeedKmph}
            onChange={(e) => setSimSpeedKmph(Number(e.target.value))}
            onMouseUp={(e) => {
              if (onSpeedChange) onSpeedChange(Number(e.target.value));
            }}
            onTouchEnd={(e) => {
              if (onSpeedChange) onSpeedChange(Number(e.target.value));
            }}
            className="w-24 accent-white cursor-pointer h-1.5"
          />
          <span className="font-bold text-white w-16">{simSpeedKmph} km/h</span>
        </div>

        {/* Bottom Right Auto Spawn Toggle */}
        {simMode === "generator" && (
          <div className="absolute bottom-3 right-3 bg-zinc-900/95 border border-zinc-700 px-3 py-1.5 rounded-md flex items-center space-x-2 text-xs font-mono text-zinc-300 shadow-md">
            <span className="text-zinc-400">Hazards (H):</span>
            <button
              onClick={() => setAutoSpawn(!autoSpawn)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold transition-colors ${autoSpawn ? "bg-emerald-600 text-white" : "bg-zinc-700 text-zinc-300"
                }`}
            >
              {autoSpawn ? "AUTO SPAWN ON" : "MANUAL ONLY"}
            </button>
          </div>
        )}
      </div>

      {/* Keyboard Shortcuts Modal */}
      {showShortcutsModal && (
        <div
          onClick={() => setShowShortcutsModal(false)}
          className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4 backdrop-blur-xs"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="bg-white border border-zinc-200 rounded-lg max-w-md w-full p-4 shadow-xl space-y-3 font-mono"
          >
            <div className="flex items-center justify-between border-b border-zinc-200 pb-2">
              <div className="flex items-center space-x-2">
                <Keyboard className="w-4 h-4 text-zinc-700" />
                <h3 className="text-xs font-bold text-zinc-900 uppercase tracking-wider">
                  Simulation Keyboard Controls
                </h3>
              </div>
              <button onClick={() => setShowShortcutsModal(false)} className="text-zinc-400 hover:text-zinc-700 transition">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold w-fit">Space</kbd>
                <span className="text-zinc-600 text-right">Pause / Resume</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold w-fit">R</kbd>
                <span className="text-zinc-600 text-right">Reset Sim & Data</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <div className="flex items-center space-x-1">
                  <kbd className="px-1.5 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">1</kbd>
                  <kbd className="px-1.5 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">2</kbd>
                  <kbd className="px-1.5 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">3</kbd>
                </div>
                <span className="text-zinc-600 text-right">Spawn Pothole / Deep / Bump</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <div className="flex items-center space-x-1">
                  <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">W</kbd>
                  <span className="text-[10px] text-zinc-400">/</span>
                  <kbd className="px-1.5 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">↑</kbd>
                </div>
                <span className="text-zinc-600 text-right">Speed Up (+5 km/h)</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <div className="flex items-center space-x-1">
                  <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">S</kbd>
                  <span className="text-[10px] text-zinc-400">/</span>
                  <kbd className="px-1.5 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold">↓</kbd>
                </div>
                <span className="text-zinc-600 text-right">Speed Down (-5 km/h)</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold w-fit">H</kbd>
                <span className="text-zinc-600 text-right">Toggle Road Hazards</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 border-b border-zinc-100 items-center">
                <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold w-fit">P</kbd>
                <span className="text-zinc-600 text-right">Cycle Road Preset</span>
              </div>

              <div className="grid grid-cols-2 py-1.5 items-center">
                <kbd className="px-2 py-0.5 bg-zinc-100 border border-zinc-300 rounded text-zinc-800 font-bold w-fit">?</kbd>
                <span className="text-zinc-600 text-right">Toggle Shortcuts Modal</span>
              </div>
            </div>

            <div className="pt-2 border-t border-zinc-100 flex justify-end">
              <button
                onClick={() => setShowShortcutsModal(false)}
                className="px-3 py-1 bg-zinc-900 text-white rounded text-xs hover:bg-zinc-800 transition"
              >
                Close (Esc)
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
