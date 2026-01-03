import React from "react";

type AlertProps = {
  type?: "error" | "success" | "info";
  children: React.ReactNode;
};

export default function Alert({ type = "info", children }: AlertProps) {
  const base = "rounded-md px-4 py-2 border flex items-start gap-3";
  if (type === "error")
    return <div className={`${base} bg-red-900/30 border-red-700 text-red-100`}>{children}</div>;
  if (type === "success")
    return <div className={`${base} bg-emerald-900/30 border-emerald-700 text-emerald-100`}>{children}</div>;
  return <div className={`${base} bg-sky-900/20 border-sky-700 text-sky-100`}>{children}</div>;
}
