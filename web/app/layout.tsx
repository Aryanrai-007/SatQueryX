import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SatQueryX · Remote Sensing Intelligence Console",
  description: "Interactive vision-language assistant for multimodal remote sensing analysis.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
