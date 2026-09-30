import React from "react";
import {
  Compass,
  Radio,
  ArrowUpDown,
  Zap,
  Thermometer,
  AlertOctagon,
  TrendingUp,
  Ruler,
} from "lucide-react";

export default function TelemetryCards({ telemetry, potholeCount, bumpCount, lastDepth }) {
  const {
    distance_cm = 0,
    baseline_cm = 0,
    deviation_cm = 0,
    strength = 0,
    temperature_c = 0,
    pi_temperature = null,
    calibrated = false,
    warmup_count = 0,
    warmup_total = 20,
  } = telemetry || {};

  const devFormatted = deviation_cm >= 0 ? `+${deviation_cm.toFixed(1)}` : `${deviation_cm.toFixed(1)}`;
  
  let devTextColor = "text-zinc-900";
  if (deviation_cm > 8.0) {
    devTextColor = "text-rose-700 font-bold";
  } else if (deviation_cm > 4.5 || deviation_cm < -4.5) {
    devTextColor = "text-amber-700 font-bold";
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
      {/* 1. Baseline */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Baseline</span>
          <Compass className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold font-mono text-zinc-950">
            {calibrated ? `${Math.round(baseline_cm)} cm` : "Warming up"}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">
            {calibrated ? "20-sample mean" : `${warmup_count}/${warmup_total} samples`}
          </p>
        </div>
      </div>

      {/* 2. Distance */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Distance</span>
          <Radio className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold font-mono text-zinc-950">
            {distance_cm > 0 ? `${distance_cm.toFixed(1)} cm` : "0.0 cm"}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Live reading</p>
        </div>
      </div>

      {/* 3. Deviation */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Deviation</span>
          <ArrowUpDown className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className={`text-xl font-mono ${devTextColor}`}>
            {devFormatted} cm
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Dist vs baseline</p>
        </div>
      </div>

      {/* 4. Signal Strength */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Strength</span>
          <Zap className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold font-mono text-zinc-950">
            {strength.toLocaleString()}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Reflectance</p>
        </div>
      </div>

      {/* 5. Temperatures */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Temperature</span>
          <Thermometer className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold font-mono text-zinc-950">
            {pi_temperature !== null ? `${pi_temperature.toFixed(1)}°C` : "--"}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Raspberry Pi</p>
          {/* <p className="text-[10px] text-zinc-500 font-mono">Sensor: {temperature_c > 0 ? `${temperature_c.toFixed(1)}°C` : "--"}</p> */}
        </div>
      </div>

      {/* 6. Potholes Count */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Potholes</span>
          <AlertOctagon className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className={`text-xl font-bold font-mono ${potholeCount > 0 ? "text-rose-700" : "text-zinc-950"}`}>
            {potholeCount}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Total detected</p>
        </div>
      </div>

      {/* 7. Speed Bumps Count */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Bumps</span>
          <TrendingUp className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className={`text-xl font-bold font-mono ${bumpCount > 0 ? "text-amber-700" : "text-zinc-950"}`}>
            {bumpCount}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Total detected</p>
        </div>
      </div>

      {/* 8. Last Depth */}
      <div className="bg-white border border-zinc-200 rounded-lg p-3 flex flex-col justify-between">
        <div className="flex items-center justify-between text-zinc-500 text-[11px] font-medium uppercase tracking-wider">
          <span>Last Depth</span>
          <Ruler className="w-3.5 h-3.5 text-zinc-400" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold font-mono text-zinc-950">
            {lastDepth > 0 ? `${lastDepth.toFixed(1)} cm` : "0.0 cm"}
          </div>
          <p className="text-[10px] text-zinc-500 font-mono mt-0.5">Max peak</p>
        </div>
      </div>
    </div>
  );
}
