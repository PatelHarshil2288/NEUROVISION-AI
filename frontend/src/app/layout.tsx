import type { Metadata } from "next";
import "./globals.css";
import "./observatory.css";
import "./training.css";
import "./explorer.css";
import "./analytics.css";

export const metadata: Metadata = {
  title: "NEUROVISION AI | Neural Network Lab",
  description:
    "Draw a digit and explore how the trained MNIST neural network is structured.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
