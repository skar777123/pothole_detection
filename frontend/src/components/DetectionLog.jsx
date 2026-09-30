import React, { useState, useMemo } from "react";
import { Download, Trash2, ListFilter, Search } from "lucide-react";

export default function DetectionLog({ logs, onClearLog }) {
  const [filterType, setFilterType] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredLogs = useMemo(() => {
    return (logs || []).filter((item) => {
      if (!item) return false;
      const itemType = String(item.type || "").toLowerCase();
      const itemTime = String(item.time || "").toLowerCase();
      const itemSev = String(item.severity || "").toLowerCase();

      const matchesType =
        filterType === "ALL" ||
        (filterType === "POTHOLE" && itemType.includes("pothole")) ||
        (filterType === "DEEP" && itemType.includes("deep")) ||
        (filterType === "BUMP" && itemType.includes("bump"));

      const query = searchQuery.trim().toLowerCase();
      const matchesQuery =
        !query ||
        itemTime.includes(query) ||
        itemType.includes(query) ||
        itemSev.includes(query);

      return matchesType && matchesQuery;
    });
  }, [logs, filterType, searchQuery]);

  const exportCsv = () => {
    if (!logs || logs.length === 0) return;
    const headers = [
      "Time",
      "Type",
      "Deviation (cm)",
      "Depth (cm)",
      "Length (cm)",
      "Width (cm)",
      "Severity",
      "Confidence",
      "Signal Strength",
      "Baseline (cm)",
    ];
    const rows = logs.map((item) => [
      item.time,
      item.type,
      item.deviation_cm,
      item.depth_cm,
      item.length_cm,
      item.width_cm,
      item.severity,
      item.confidence,
      item.strength,
      item.baseline,
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `pothole_detection_log_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white border border-zinc-200 rounded-lg overflow-hidden">
      {/* Top Header & Filter Controls */}
      <div className="px-5 py-3.5 border-b border-zinc-200 flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <ListFilter className="w-4 h-4 text-zinc-700" />
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-900">
            Anomaly Detection Event Log
          </h2>
          <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-100 text-zinc-700 border border-zinc-200 font-mono font-medium">
            {logs.length} events
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Quick Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-zinc-400" />
            <input
              type="text"
              placeholder="Filter logs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-zinc-50 text-zinc-900 text-xs rounded-md pl-8 pr-3 py-1 border border-zinc-300 focus:outline-none focus:border-zinc-500 w-36 sm:w-44"
            />
          </div>

          {/* Type filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="bg-white text-zinc-800 text-xs rounded-md px-2.5 py-1 border border-zinc-300 focus:outline-none focus:border-zinc-500"
          >
            <option value="ALL">All Types</option>
            <option value="POTHOLE">All Potholes</option>
            <option value="DEEP">Deep Potholes Only</option>
            <option value="BUMP">Speed Bumps</option>
          </select>

          {/* Action buttons */}
          <button
            onClick={exportCsv}
            disabled={logs.length === 0}
            className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-zinc-50 border border-zinc-300 transition disabled:opacity-40"
          >
            <Download className="w-3 h-3" />
            <span>Export CSV</span>
          </button>

          <button
            onClick={onClearLog}
            disabled={logs.length === 0}
            className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-md text-xs font-medium bg-white text-zinc-700 hover:bg-rose-50 hover:text-rose-700 hover:border-rose-200 border border-zinc-300 transition disabled:opacity-40"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Table view */}
      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        {filteredLogs.length === 0 ? (
          <div className="py-12 text-center text-xs text-zinc-500 font-mono">
            {logs.length === 0
              ? "No road surface anomalies detected in current session."
              : "No events match current filter."}
          </div>
        ) : (
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-50 text-zinc-600 border-b border-zinc-200 uppercase tracking-wider text-[10px] font-semibold sticky top-0">
              <tr>
                <th className="py-2.5 px-4">Time</th>
                <th className="py-2.5 px-4">Type</th>
                <th className="py-2.5 px-4">Dev (cm)</th>
                <th className="py-2.5 px-4">Depth (cm)</th>
                <th className="py-2.5 px-4">Length (cm)</th>
                <th className="py-2.5 px-4">Width (cm)</th>
                <th className="py-2.5 px-4">Severity</th>
                <th className="py-2.5 px-4">Confidence</th>
                <th className="py-2.5 px-4">Strength</th>
                <th className="py-2.5 px-4">Baseline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 font-mono">
              {filteredLogs.map((row, index) => {
                const rType = String(row?.type || "").toLowerCase();
                const rSev = String(row?.severity || "").toLowerCase();
                const isDeep = rType.includes("deep") || rSev.includes("deep");
                const isPothole = rType.includes("pothole");

                let badgeClass = "bg-zinc-100 text-zinc-800 border-zinc-200";
                if (isDeep) {
                  badgeClass = "bg-rose-50 text-rose-700 border-rose-200 font-bold";
                } else if (isPothole) {
                  badgeClass = "bg-amber-50 text-amber-800 border-amber-200 font-semibold";
                } else {
                  badgeClass = "bg-amber-50 text-amber-900 border-amber-200";
                }

                return (
                  <tr
                    key={row.id || index}
                    className="hover:bg-zinc-50 transition-colors"
                  >
                    <td className="py-2 px-4 text-zinc-500">{row.time}</td>
                    <td className="py-2 px-4">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[11px] border ${badgeClass}`}
                      >
                        {row.type}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-zinc-900 font-semibold">
                      {row.deviation_cm}
                    </td>
                    <td className="py-2 px-4 font-bold text-zinc-950">
                      {row.depth_cm}
                    </td>
                    <td className="py-2 px-4 text-zinc-700">
                      {row.length_cm}
                    </td>
                    <td className="py-2 px-4 text-zinc-700">
                      {row.width_cm}
                    </td>
                    <td className="py-2 px-4">
                      <span
                        className={
                          isDeep
                            ? "text-rose-700 font-semibold"
                            : row.severity.toLowerCase().includes("moderate")
                            ? "text-amber-800"
                            : "text-zinc-600"
                        }
                      >
                        {row.severity}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-zinc-600">
                      {row.confidence}
                    </td>
                    <td className="py-2 px-4 text-zinc-600">
                      {row.strength}
                    </td>
                    <td className="py-2 px-4 text-zinc-500">
                      {row.baseline} cm
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
