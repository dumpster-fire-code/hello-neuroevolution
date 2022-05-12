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
  mutationRate = 0.002;
  numGenerations = 0;
  numMovesRemaining = 0;
  population: Agent[] = [];
  populationSize = 75;
  topPerformersPct = 0.65;
  elitePerformersPct = 0.04;

  nextGeneration() {
    const map = this.buildMap();

    if (this.numGenerations === 0) {
      this.population = times(this.populationSize).map(() =>
        this.buildAgent({ map, model: this.buildModel(null) }),
      );
    } else {
      const prevPopulation = this.population;
      const nextPopulation: Agent[] = [];

      this.population.sort((a, b) => b.score - a.score);

      const numElitePerformers = Math.round(
        this.populationSize * this.elitePerformersPct,
      );
      const numTopPerformers = Math.round(
        this.populationSize * this.topPerformersPct,
      );

      const elitePerformers = this.population.slice(0, numElitePerformers);
      const topPerformers = this.population.slice(0, numTopPerformers);

      elitePerformers.forEach((elitePerformer) => {
        const model = this.buildModel(elitePerformer.model.getWeights());
        nextPopulation.push(this.buildAgent({ map, model }));
      });

      while (nextPopulation.length < this.populationSize) {
        const weights = this.buildWeights(
          this.selectParent(sample(topPerformers)!, sample(topPerformers)!),
          this.selectParent(sample(topPerformers)!, sample(topPerformers)!),
        );

        const model = this.buildModel(weights);
        nextPopulation.push(this.buildAgent({ map, model }));
      }

      // console.log('~~~~~~~~~~~~~~~~~~~~~~~~~~~');
      // console.log(
      //   'average elite performer score:',
      //   elitePerformers.reduce((sum, { score }) => sum + score, 0) /
      //     elitePerformers.length,
      // );
      // console.log(
      //   'average top performer score:',
      //   topPerformers.reduce((sum, { score }) => sum + score, 0) /
      //     topPerformers.length,
      // );
      // console.log(
      //   'average  score:',
      //   this.population.reduce((sum, { score }) => sum + score, 0) /
      //     this.population.length,
      // );

      this.population = nextPopulation;

      prevPopulation.forEach((agent) => {
        agent.model.dispose();
      });
    }

    this.numGenerations++;
    this.numMovesRemaining = this.movesPerGeneration;
  }

  update() {
    this.numMovesRemaining--;

    this.population.forEach((agent) => {
      const {
        position,
        map: { end, nodes },
      } = agent;

      if (position.x === end.x && position.y === end.y) {
        return;
      }

      const nextMove = this.getNextMove(agent);
      const offset = offsets[nextMove];

      const possibleNextPosition: Position = {
        x: position.x + offset.x,
        y: position.y + offset.y,
      };

      agent.numMoves++;
      agent.directionsMoved.add(nextMove);

      if (
        this.isObstructionAt(
          nodes,
          possibleNextPosition.x,
          possibleNextPosition.y,
        )
      ) {
        agent.numCollisions++;
      } else {
        nodes[possibleNextPosition.x][possibleNextPosition.y].visitCount++;
        agent.position = possibleNextPosition;
      }
    });

    this.population.forEach((agent) => {
      this.updateScore(agent);
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
      labeledInputs: {},
      labeledScoringFactors: {},
      map: cloneDeep(map),
      model,
      numCollisions: 0,
      numMoves: 0,
      position: { ...map.start },
      score: 0,
    };
  }

  private buildWeights(parent1: Agent, parent2: Agent): Tensor[] {
    return tf.tidy(() => {
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
    });
  }

  private buildModel(weights: Tensor[] | null): tf.Sequential {
    const model = tf.sequential();

    if (weights) {
      model.setFastWeightInitDuringBuild(true);
    }

    // model.add(
    //   tf.layers.dense({
    //     name: 'hidden',
    //     units: 6, // https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3 "Most of the problems can be solved by using a single hidden layer with the number of neurons equal to the mean of the input and output layer."
    //     inputShape: [10],
    //     activation: 'tanh',
    //   }),
    // );

    model.add(
      tf.layers.dense({
        name: 'output',
        units: 4,
        inputShape: [12],
        activation: 'softmax',
      }),
    );

    if (weights) {
      model.setWeights(weights);
    }

    return model;
  }

  private getNextMove(agent: Agent): Direction {
    const {
      model,
      map: { end, nodes },
      position: { x, y },
    } = agent;

    return tf.tidy(() => {
      const unexploredInputs = this.normalize([
        this.getUnexploredInput(nodes, x, y - 1), // up
        this.getUnexploredInput(nodes, x + 1, y), // right
        this.getUnexploredInput(nodes, x, y + 1), // down
        this.getUnexploredInput(nodes, x - 1, y), // left
      ]);

      const labeledInputs = {
        'available ↑': this.getAvailableNeighborInput(nodes, x, y - 1),
        'available →': this.getAvailableNeighborInput(nodes, x + 1, y),
        'available ↓': this.getAvailableNeighborInput(nodes, x, y + 1),
        'available ←': this.getAvailableNeighborInput(nodes, x - 1, y),
        'unexplored ↑': unexploredInputs[0],
        'unexplored →': unexploredInputs[1],
        'unexplored ↓': unexploredInputs[2],
        'unexplored ←': unexploredInputs[3],
        'goal ↔': (end.x - x) / this.mapSize,
        'goal ↕': (end.y - y) / this.mapSize,
      };

      agent.labeledInputs = labeledInputs;

      const tfInputs = tf.tensor2d([
        (Object.values(labeledInputs) as number[]).concat([
          labeledInputs['goal ↔'],
          labeledInputs['goal ↕'],
        ]),
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

  private updateScore(agent: Agent) {
    const {
      map: { start, end, nodes },
      numCollisions,
      numMoves,
      position,
    } = agent;

    const startToEndDistance = this.computeAbsoluteDistance(start, end);
    const positionToEndDistance = this.computeAbsoluteDistance(position, end);
    const distancePct = 1 - positionToEndDistance / startToEndDistance;
    const pctMovesUsed = numMoves / this.movesPerGeneration;

    let backtracks = 0;

    for (let x = 0; x < this.mapSize; x++) {
      for (let y = 0; y < this.mapSize; y++) {
        backtracks += Math.max(0, nodes[x][y].visitCount - 2);
      }
    }

    const progress = (5 * distancePct) / pctMovesUsed;
    const collisionsPenalty = numCollisions;
    const backtrackPenalty = backtracks;
    const reachedEndBonus = distancePct === 1 ? 10 : 0;

    const score =
      progress - collisionsPenalty - backtrackPenalty + reachedEndBonus;

    agent.labeledScoringFactors = {
      progress,
      collisions: collisionsPenalty,
      backtracks: backtrackPenalty,
      reachedEnd: reachedEndBonus,
      score,
    };

    agent.score = score;
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
          const isObstruction = !isStart && !isEnd && random(true) < 0.2;
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
    const node = this.getNode(nodes, x, y);

    if (!node) {
      return true;
    }

    return node.isObstruction;
  }

  private getNode(
    nodes: MapNode[][],
    x: number,
    y: number,
  ): MapNode | undefined {
    if (nodes[x] === undefined) {
      return;
    }

    return nodes[x][y];
  }

  private getAvailableNeighborInput(
    nodes: MapNode[][],
    x: number,
    y: number,
  ): number {
    const isObstructed = this.isObstructionAt(nodes, x, y);
    return isObstructed ? -1 : 1;
  }

  private getUnexploredInput(nodes: MapNode[][], x: number, y: number): number {
    const node = this.getNode(nodes, x, y);

    if (!node || this.isObstructionAt(nodes, x, y)) {
      return -1;
    }

    const neighborNodesBonus =
      [
        this.getNode(nodes, x, y - 1), // up
        this.getNode(nodes, x + 1, y), // right
        this.getNode(nodes, x, y + 1), // down
        this.getNode(nodes, x - 1, y), // left
      ].filter(
        (adjacentNode) =>
          adjacentNode &&
          !adjacentNode.isObstruction &&
          adjacentNode.visitCount === 0,
      ).length * 0.25;

    return (
      (() => {
        switch (node.visitCount) {
          case 0:
            return 1;
          case 1:
            return 0;
          case 2:
            return -0.5;
          default:
            return -1;
        }
      })() + neighborNodesBonus
    );
  }

  private normalize(values: number[]): number[] {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;

    return values.map((value) => {
      if (range === 0) {
        return 0;
      }

      return (2 * (value - min)) / range - 1;
    });
  }

  private selectParent(candidate1: Agent, candidate2: Agent) {
    return candidate1.score >= candidate2.score ? candidate1 : candidate2;
  }
}

export { Engine };
