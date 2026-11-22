import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "3D Neural Observatory | NEUROVISION AI",
  description:
    "Explore the 986-neuron MNIST network in an interactive, performance-aware 3D view.",
};

export default function ObservatoryLayout({
  children,
}: {
  children: ReactNode;
}) {
  return children;
}
