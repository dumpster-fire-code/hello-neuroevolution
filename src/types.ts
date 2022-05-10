import * as tf from '@tensorflow/tfjs';

export interface Map {
  end: Position;
  nodes: MapNode[][];
  start: Position;
}

export interface MapNode {
  isObstruction: boolean;
  visitCount: number;
}

export interface Agent {
  directionsMoved: Set<Direction>;
  id: string;
  map: Map;
  model: tf.Sequential;
  numCollisions: number;
  numMoves: number;
  position: Position;
}

export interface Position {
  x: number;
  y: number;
}

export const directions = ['up', 'right', 'down', 'left'] as const;

export type Direction = typeof directions[number];
