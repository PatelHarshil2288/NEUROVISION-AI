"use client";

import dynamic from "next/dynamic";

const NeuralObservatory = dynamic(
  () =>
    import("@/features/neural-observatory/components/NeuralObservatory"),
  {
    ssr: false,
    loading: () => (
      <main className="observatory-route-loading" role="status">
        Loading the Neural Observatory…
      </main>
    ),
  },
);

export default function ObservatoryPage() {
  return <NeuralObservatory />;
}
