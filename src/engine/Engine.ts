import * as tf from '@tensorflow/tfjs';
import { Tensor } from '@tensorflow/tfjs';
import cloneDeep from 'lodash/cloneDeep';
import random from 'lodash/random';
import sample from 'lodash/sample';
import times from 'lodash/times';
import uniqueId from 'lodash/uniqueId';
import { Agent, Direction, directions, Map, MapNode, Position } from 'types';

const offsets: Record<Direction, Position> = {
  up: { x: 0, y: -1 },
  right: { x: 1, y: 0 },
  down: { x: 0, y: 1 },
  left: { x: -1, y: 0 },
};

tf.setBackend('cpu');

class Engine {
  bestScoreAllTime = 0;
  bestScoreGenerational = 0;
  mapSize = 20;
  movesPerGeneration = 50;
  mutationRate = 0.005;
  numGenerations = 0;
  numMovesRemaining = 0;
  population: Agent[] = [];
  populationSize = 50;

  nextGeneration() {
    const map = this.buildMap();

    if (this.numGenerations === 0) {
      this.population = times(this.populationSize).map(() =>
        this.buildAgent({ map, model: this.buildModel(null) }),
      );
    } else {
      this.population = times(this.populationSize).map(() => {
        const weights = this.buildWeights(
          this.selectParent(sample(this.population)!, sample(this.population)!),
          this.selectParent(sample(this.population)!, sample(this.population)!),
        );

        const model = this.buildModel(weights);

        return this.buildAgent({ map, model });
      });
    }

    this.numGenerations++;
    this.numMovesRemaining = this.movesPerGeneration;
  }

  update() {
    this.numMovesRemaining--;

    this.population.forEach((agent) => {
      if (
        agent.position.x === agent.map.end.x &&
        agent.position.y === agent.map.end.y
      ) {
        return;
      }

      const nextMove = this.getNextMove(agent);
      const offset = offsets[nextMove];

      const possibleNextPosition: Position = {
        x: agent.position.x + offset.x,
        y: agent.position.y + offset.y,
      };

      agent.numMoves++;
      agent.directionsMoved.add(nextMove);

      if (
        possibleNextPosition.x < 0 ||
        possibleNextPosition.x > this.mapSize - 1 ||
        possibleNextPosition.y < 0 ||
        possibleNextPosition.y > this.mapSize - 1 ||
        this.isObstructionAt(
          agent.map.nodes,
          possibleNextPosition.x,
          possibleNextPosition.y,
        )
      ) {
        agent.numCollisions++;
      } else {
        agent.map.nodes[possibleNextPosition.x][possibleNextPosition.y]
          .visitCount++;
        agent.position = possibleNextPosition;
      }
    });
  }

  private buildAgent({
    map,
    model,
  }: {
    map: Map;
    model: tf.Sequential;
  }): Agent {
    return {
      directionsMoved: new Set(),
      id: uniqueId('agent_'),
      model,
      numCollisions: 0,
      numMoves: 0,
      position: { ...map.start },
      map: cloneDeep(map),
    };
  }

  private buildWeights(parent1: Agent, parent2: Agent): Tensor[] {
    const parent1WrappedWeights = parent1.model.getWeights();
    const parent2WrappedWeights = parent2.model.getWeights();

    const weights: number[][] = [];

    for (let i = 0; i < parent1WrappedWeights.length; i++) {
      weights[i] = [];

      const parent1Weights = parent1WrappedWeights[i].dataSync();
      const parent2Weights = parent2WrappedWeights[i].dataSync();

      for (let j = 0; j < parent1Weights.length; j++) {
        if (random(true) < this.mutationRate) {
          weights[i].push(random());
        } else {
          const parent1Weight = parent1Weights[j];
          const parent2Weight = parent2Weights[j];
          weights[i].push(sample([parent1Weight, parent2Weight])!);
        }
      }
    }

    const weightsWrapped: Tensor[] = [];

    for (let i = 0; i < weights.length; i++) {
      weightsWrapped.push(
        tf.tensor(weights[i], parent1WrappedWeights[i].shape),
      );
    }

    return weightsWrapped;
  }

  private buildModel(weights: Tensor[] | null): tf.Sequential {
    const model = tf.sequential();

    if (weights) {
      model.setFastWeightInitDuringBuild(true);
    }

    model.add(
      tf.layers.dense({
        name: 'hidden',
        units: 2,
        inputShape: [2],
        // units: 7, // https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3 "Most of the problems can be solved by using a single hidden layer with the number of neurons equal to the mean of the input and output layer."
        // inputShape: [10], // obstructions for each side, visited count each side, x/y distance from end
        activation: 'tanh',
      }),
    );

    model.add(
      tf.layers.dense({
        name: 'output',
        units: 4,
        activation: 'softmax',
      }),
    );

    if (weights) {
      model.setWeights(weights);
    }

    return model;
  }

  private getNextMove({
    model,
    map: { end, nodes },
    numMoves,
    position: { x, y },
  }: Agent): Direction {
    return tf.tidy(() => {
      const tfInputs = tf.tensor2d([
        [
          // this.boolToNum(this.isObstructionAt(nodes, x, y - 1)), // up
          // this.boolToNum(this.isObstructionAt(nodes, x + 1, y)), // right
          // this.boolToNum(this.isObstructionAt(nodes, x, y + 1)), // down
          // this.boolToNum(this.isObstructionAt(nodes, x - 1, y)), // left
          // this.getNormalizedVisitCount(nodes, numMoves, x, y - 1), // up
          // this.getNormalizedVisitCount(nodes, numMoves, x + 1, y), // right
          // this.getNormalizedVisitCount(nodes, numMoves, x, y + 1), // down
          // this.getNormalizedVisitCount(nodes, numMoves, x - 1, y), // left
          (end.x - x) / this.mapSize,
          (end.y - y) / this.mapSize,
        ],
      ]);
      const tfOutputs = model.predict(tfInputs) as tf.Tensor;
      const outputs = tfOutputs.dataSync() as unknown as number[];

      switch (Math.max(...outputs)) {
        case outputs[0]:
          return 'up';
        case outputs[1]:
          return 'right';
        case outputs[2]:
          return 'down';
        case outputs[3]:
          return 'left';
      }

      throw new Error(`Could not map outputs to direction: ${outputs}`);
    });
  }

  private computeScore({
    // directionsMoved,
    map: { start, end },
    // numCollisions,
    numMoves,
    position,
  }: Agent) {
    const startToEndDistance = this.computeAbsoluteDistance(start, end);
    const positionToEndDistance = this.computeAbsoluteDistance(position, end);
    const distancePctProgress = 1 - positionToEndDistance / startToEndDistance;
    const pctMovesUsed = numMoves / this.movesPerGeneration;

    return distancePctProgress / pctMovesUsed;
  }

  private computeAbsoluteDistance(
    { x: p1x, y: p1y }: Position,
    { x: p2x, y: p2y }: Position,
  ) {
    return Math.abs(p1x - p2x) + Math.abs(p1y - p2y);
  }

  private buildMap(): Map {
    let map: Map;

    do {
      map = ((): Map => {
        const direction = sample(directions);
        console.log(direction);
        switch (direction) {
          // Start at bottom.
          case 'up':
            return {
              nodes: [[]],
              start: { x: random(this.mapSize - 1), y: this.mapSize - 1 },
              end: { x: random(this.mapSize - 1), y: 0 },
            };
          // Start at top.
          case 'down':
            return {
              nodes: [[]],
              start: { x: random(this.mapSize - 1), y: 0 },
              end: { x: random(this.mapSize - 1), y: this.mapSize - 1 },
            };
          // Start from left.
          case 'left':
            return {
              nodes: [[]],
              start: { x: 0, y: random(this.mapSize - 1) },
              end: { x: this.mapSize - 1, y: random(this.mapSize - 1) },
            };
          // Start from right.
          default:
            return {
              nodes: [[]],
              start: { x: this.mapSize - 1, y: random(this.mapSize - 1) },
              end: { x: 0, y: random(this.mapSize - 1) },
            };
        }
      })();

      for (let x = 0; x < this.mapSize; x++) {
        map.nodes[x] = [];

        for (let y = 0; y < this.mapSize; y++) {
          const isStart = map.start.x === x && map.start.y === y;
          const isEnd = map.end.x === x && map.end.y === y;
          const isObstruction = !isStart && !isEnd && false; // && random(0.2) === 0;
          map.nodes[x][y] = { isObstruction, visitCount: isStart ? 1 : 0 };
        }
      }
    } while (!this.isMapValid(map));

    return map;
  }

  private isMapValid({ nodes, start, end }: Map) {
    const queue: Position[] = [start];
    const seen: boolean[][] = times(this.mapSize).map(() => []);

    const addToQueue = ({ x, y }: Position) => {
      if (
        x >= 0 &&
        x < this.mapSize &&
        y >= 0 &&
        y < this.mapSize &&
        !this.isObstructionAt(nodes, x, y) &&
        !seen[x][y]
      ) {
        seen[x][y] = true;
        queue.push({ x, y });
      }
    };

    while (queue.length > 0) {
      const { x, y } = queue.shift()!;
      if (x === end.x && y === end.y) {
        return true;
      }

      addToQueue({ x: x - 1, y });
      addToQueue({ x: x + 1, y });
      addToQueue({ x, y: y - 1 });
      addToQueue({ x, y: y + 1 });
    }

    return false;
  }

  private isObstructionAt(nodes: MapNode[][], x: number, y: number): boolean {
    const node = this.getMapNode(nodes, x, y);

    if (!node) {
      return true;
    }

    return node.isObstruction;
  }

  private getNormalizedVisitCount(
    nodes: MapNode[][],
    numMoves: number,
    x: number,
    y: number,
  ): number {
    const node = this.getMapNode(nodes, x, y);

    if (!node || !numMoves) {
      return 0;
    }

    return node.visitCount / numMoves;
  }

  private getMapNode(
    nodes: MapNode[][],
    x: number,
    y: number,
  ): MapNode | undefined {
    if (nodes[x] === undefined) {
      return;
    }

    return nodes[x][y];
  }

  private boolToNum(value: boolean) {
    return value ? 1 : 0;
  }

  private selectParent(candidate1: Agent, candidate2: Agent) {
    return this.computeScore(candidate1) >= this.computeScore(candidate2)
      ? candidate1
      : candidate2;
  }
}

export { Engine };
