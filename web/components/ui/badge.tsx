import * as React from "react";
import { cn } from "@/lib/utils";

export function Badge({ className, children, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("inline-flex items-center gap-1 rounded-full border border-cyan-300/15 bg-cyan-300/[0.06] px-2.5 py-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-cyan-100", className)} {...props}>{children}</span>;
}
