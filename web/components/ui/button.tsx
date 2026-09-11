import * as React from "react";
import { cn } from "@/lib/utils";

export function Button({ className, variant = "default", size = "default", ...props }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "ghost" | "outline"; size?: "default" | "sm" }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all disabled:pointer-events-none disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-cyan-400/30",
        variant === "default" && "bg-cyan-300 text-slate-950 hover:bg-cyan-200 shadow-lg shadow-cyan-500/10",
        variant === "ghost" && "text-slate-400 hover:bg-white/[0.05] hover:text-white",
        variant === "outline" && "border border-white/10 bg-white/[0.025] text-slate-200 hover:border-cyan-300/30 hover:bg-cyan-300/[0.06]",
        size === "default" ? "h-10 px-4 text-sm" : "h-8 px-3 text-xs",
        className,
      )}
      {...props}
    />
  );
}
