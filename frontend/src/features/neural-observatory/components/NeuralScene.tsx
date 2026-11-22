"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, type ComponentRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

import {
  NETWORK_LAYERS,
  NETWORK_TOPOLOGY,
  type LayerId,
} from "../lib/network";
import {
  getConnectionContributions,
  getLayerDisplayIntensities,
  type NeuronSelection,
} from "../lib/signal";
import type {
  ActivationsResult,
  ModelWeightsResult,
} from "../../digit-studio/lib/api";

interface NeuralSceneProps {
  focusedLayerId: LayerId | null;
  resetSignal: number;
  activations: ActivationsResult | null;
  weights: ModelWeightsResult | null;
  activeLayerIndex: number;
  selectedNeuron: NeuronSelection | null;
  onSelectNeuron: (selection: NeuronSelection) => void;
}

const EDGE_COLORS = NETWORK_LAYERS.slice(0, -1).map((layer) => layer.color);

function NetworkGeometry({
  focusedLayerId,
  activations,
  weights,
  activeLayerIndex,
  selectedNeuron,
  onSelectNeuron,
}: {
  focusedLayerId: LayerId | null;
  activations: ActivationsResult | null;
  weights: ModelWeightsResult | null;
  activeLayerIndex: number;
  selectedNeuron: NeuronSelection | null;
  onSelectNeuron: (selection: NeuronSelection) => void;
}) {
  const neuronsRef = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((state) => state.invalidate);
  const sphereGeometry = useMemo(
    () => new THREE.SphereGeometry(1, 8, 8),
    [],
  );
  const sphereMaterial = useMemo(
    () => new THREE.MeshBasicMaterial({ toneMapped: false }),
    [],
  );
  const layerIntensities = useMemo(
    () =>
      activations
        ? {
            input: getLayerDisplayIntensities(activations, "input"),
            hidden_1: getLayerDisplayIntensities(activations, "hidden_1"),
            hidden_2: getLayerDisplayIntensities(activations, "hidden_2"),
            output: getLayerDisplayIntensities(activations, "output"),
          }
        : null,
    [activations],
  );
  const connectionGeometry = useMemo(() => {
    const positions = new Float32Array(NETWORK_TOPOLOGY.connections.length * 6);
    const colors = new Float32Array(NETWORK_TOPOLOGY.connections.length * 6);
    const focusedIndex = focusedLayerId
      ? NETWORK_LAYERS.findIndex((layer) => layer.id === focusedLayerId)
      : -1;
    const connectionSignals =
      activations && weights
        ? (
            [
              getConnectionContributions(activations, weights, 0),
              getConnectionContributions(activations, weights, 1),
              getConnectionContributions(activations, weights, 2),
            ] as const
          ).flat()
        : null;
    const maxContributionByTransition = [0, 1, 2].map((transition) =>
      connectionSignals
        ? Math.max(
            0,
            ...connectionSignals
              .filter((signal) => signal.transition === transition)
              .map((signal) => Math.abs(signal.contribution)),
          )
        : 0,
    );

    NETWORK_TOPOLOGY.connections.forEach((connection, index) => {
      const source = NETWORK_TOPOLOGY.neurons[connection.from];
      const target = NETWORK_TOPOLOGY.neurons[connection.to];
      if (!source || !target) {
        throw new Error("A sampled connection references an unknown neuron.");
      }

      positions.set(source.position, index * 6);
      positions.set(target.position, index * 6 + 3);

      const color = new THREE.Color(EDGE_COLORS[connection.transition]);
      if (connectionSignals) {
        const signal = connectionSignals[index];
        if (!signal) {
          throw new Error("The sampled connection signals do not match topology.");
        }
        color.set(
          connection.transition < activeLayerIndex
            ? signal.contribution < 0
              ? "#f28aa8"
              : "#66e4e6"
            : "#263342",
        );
        const maximum =
          maxContributionByTransition[connection.transition] ?? 0;
        const intensity =
          maximum > 0
            ? Math.abs(signal.contribution) / maximum
            : 0;
        color.lerp(
          new THREE.Color("#18222e"),
          0.82 * (1 - intensity),
        );
      }
      if (
        focusedIndex >= 0 &&
        connection.transition !== focusedIndex &&
        connection.transition !== focusedIndex - 1
      ) {
        color.multiplyScalar(0.22);
      }
      color.toArray(colors, index * 6);
      color.toArray(colors, index * 6 + 3);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();
    return geometry;
  }, [
    activeLayerIndex,
    activations,
    focusedLayerId,
    weights,
  ]);

  useLayoutEffect(() => {
    const mesh = neuronsRef.current;
    if (!mesh) {
      return;
    }

    const transform = new THREE.Object3D();
    const color = new THREE.Color();
    for (let index = 0; index < NETWORK_TOPOLOGY.neurons.length; index += 1) {
      const neuron = NETWORK_TOPOLOGY.neurons[index];
      if (!neuron) {
        throw new Error("A network neuron is missing from the topology.");
      }
      transform.position.set(...neuron.position);
      const selected = neuron.layerId === selectedNeuron?.layerId && neuron.index === selectedNeuron.index;
      transform.scale.setScalar(neuron.radius * (selected ? 1.8 : 1));
      transform.updateMatrix();
      mesh.setMatrixAt(index, transform.matrix);

      color.set(neuron.color);
      if (layerIntensities) {
        const intensity = layerIntensities[neuron.layerId][neuron.index];
        if (intensity === undefined) {
          throw new Error("The activation tensor does not match its network layer.");
        }
        color.lerp(
          new THREE.Color("#101722"),
          0.86 * (1 - intensity),
        );
        const layerIndex = NETWORK_LAYERS.findIndex(
          (layer) => layer.id === neuron.layerId,
        );
        if (layerIndex > activeLayerIndex) {
          color.multiplyScalar(0.18);
        }
      }
      if (focusedLayerId && neuron.layerId !== focusedLayerId) {
        color.multiplyScalar(0.32);
      }
      if (selected) {
        color.set("#ffffff");
      }
      mesh.setColorAt(index, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
    mesh.computeBoundingSphere();
    invalidate();
  }, [activeLayerIndex, focusedLayerId, invalidate, layerIntensities, selectedNeuron]);

  useEffect(() => () => connectionGeometry.dispose(), [connectionGeometry]);

  return (
    <>
      <instancedMesh
        ref={neuronsRef}
        args={[
          sphereGeometry,
          sphereMaterial,
          NETWORK_TOPOLOGY.neurons.length,
        ]}
        frustumCulled={false}
        onClick={(event) => {
          event.stopPropagation();
          const neuron = event.instanceId === undefined ? undefined : NETWORK_TOPOLOGY.neurons[event.instanceId];
          if (neuron) {
            onSelectNeuron({ layerId: neuron.layerId, index: neuron.index });
          }
        }}
      />
      <lineSegments geometry={connectionGeometry} frustumCulled={false}>
        <lineBasicMaterial
          transparent
          opacity={0.25}
          vertexColors
          depthWrite={false}
        />
      </lineSegments>
    </>
  );
}

export default function NeuralScene({
  focusedLayerId,
  resetSignal,
  activations,
  weights,
  activeLayerIndex,
  selectedNeuron,
  onSelectNeuron,
}: NeuralSceneProps) {
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null);
  const previousFocus = useRef<LayerId | null>(null);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) {
      return;
    }

    if (focusedLayerId) {
      const layer = NETWORK_LAYERS.find(
        (candidate) => candidate.id === focusedLayerId,
      );
      if (!layer) {
        return;
      }
      controls.target.set(0, 0, layer.positionZ);
      controls.object.position.set(0, 0, layer.positionZ + 14);
      controls.update();
    } else if (previousFocus.current || resetSignal > 0) {
      controls.reset();
    }
    previousFocus.current = focusedLayerId;
  }, [focusedLayerId, resetSignal]);

  return (
    <Canvas
      aria-label="Interactive 3D structural view of the 784, 128, 64, and 10 neuron layers"
      camera={{ position: [-17, 11, 25], fov: 42, near: 0.1, far: 100 }}
      dpr={[1, 1.4]}
      fallback={
        <div className="observatory-webgl-fallback" role="status">
          This browser could not start WebGL. Try a browser with WebGL enabled.
        </div>
      }
      frameloop="demand"
      gl={{ alpha: true, antialias: false, powerPreference: "low-power" }}
    >
      <NetworkGeometry
        focusedLayerId={focusedLayerId}
        activations={activations}
        weights={weights}
        activeLayerIndex={activeLayerIndex}
        selectedNeuron={selectedNeuron}
        onSelectNeuron={onSelectNeuron}
      />
      <OrbitControls
        ref={controlsRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={11}
        maxDistance={46}
        maxPolarAngle={Math.PI}
        minPolarAngle={0}
      />
    </Canvas>
  );
}
