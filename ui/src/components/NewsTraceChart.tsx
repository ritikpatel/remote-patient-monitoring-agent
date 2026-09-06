import type { TracePoint } from "../types";

/** Hand-rolled SVG line chart for the NEWS2 trace (PROJECT_PLAN.md section
 * 12: "live vitals with the NEWS2 trace"). No charting library dependency --
 * a NEWS2 score is a small integer series (0-20-ish) plotted against hour,
 * which a plain polyline renders exactly as well as a library would, for a
 * fraction of the bundle size and zero new untested third-party code. */
export function NewsTraceChart({ points }: { points: TracePoint[] }) {
  if (points.length === 0) return <p className="muted">No trace data.</p>;

  const width = 640;
  const height = 180;
  const padding = 28;
  const maxNews2 = Math.max(20, ...points.map((p) => p.news2));
  const minHour = points[0].hour;
  const maxHour = points[points.length - 1].hour;
  const hourSpan = Math.max(1, maxHour - minHour);

  const x = (hour: number) => padding + ((hour - minHour) / hourSpan) * (width - 2 * padding);
  const y = (news2: number) => height - padding - (news2 / maxNews2) * (height - 2 * padding);

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.hour)} ${y(p.news2)}`).join(" ");

  // Threshold guides at the recalibrated ICU tiers this dashboard reasons
  // about everywhere else (warehouse/news2.py's ICU_TIER_THRESHOLDS).
  const mediumY = y(5);
  const highY = y(7);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="news-trace-chart"
      role="img"
      aria-label="NEWS2 score over the stay"
    >
      <line x1={padding} y1={mediumY} x2={width - padding} y2={mediumY} className="trace-guide trace-guide--medium" />
      <line x1={padding} y1={highY} x2={width - padding} y2={highY} className="trace-guide trace-guide--high" />
      <path d={linePath} className="trace-line" fill="none" />
      {points.map((p) => (
        <circle
          key={p.hour}
          cx={x(p.hour)}
          cy={y(p.news2)}
          r={3}
          className={`trace-point trace-point--${p.news2_tier_icu}`}
        >
          <title>{`hour ${p.hour}: NEWS2 ${p.news2} (${p.news2_tier_icu})`}</title>
        </circle>
      ))}
      <text x={padding} y={height - 6} className="trace-axis-label">
        hour {minHour}
      </text>
      <text x={width - padding} y={height - 6} textAnchor="end" className="trace-axis-label">
        hour {maxHour}
      </text>
    </svg>
  );
}
