import type { Tier } from "../types";

/** Colour-banded escalation tier (PROJECT_PLAN.md section 12: "colour-banded
 * by escalation tier"). Text label always accompanies colour -- colour alone
 * is not an accessible way to convey a clinical tier. */
export function RiskBadge({ tier }: { tier: Tier | null | undefined }) {
  const t = tier ?? "low";
  return <span className={`risk-badge risk-badge--${t}`}>{t.toUpperCase()}</span>;
}
