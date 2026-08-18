import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aporia — search philosophy by argument, not keyword",
  description:
    "Type a claim. See which philosophers argued for it and which argued against it, " +
    "across 2,000 years of primary sources.",
  openGraph: {
    title: "Aporia — search philosophy by argument, not keyword",
    description:
      "Semantic search over primary-source philosophy that classifies whether a passage " +
      "argues for or against a claim, not merely whether it is about the topic.",
    type: "website",
  },
};

export const viewport: Viewport = {
  // Both are declared so the browser picks the matching UI chrome instead of
  // defaulting to light and flashing white before the CSS applies.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#faf7f2" },
    { media: "(prefers-color-scheme: dark)", color: "#16140f" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-dvh antialiased">{children}</body>
    </html>
  );
}
