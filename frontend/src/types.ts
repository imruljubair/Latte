export type Tensor = number[] | number[][];
export type Weights = Record<string, Tensor>;
export interface CorpusUsage {
  ranges: number[][];
  max_count: number;
  used_positions: number;
  total_uses: number;
}
export interface Step {
  corpus_usage?: CorpusUsage;
  step: number;
  loss: number;
  starts: number[];
  inputs: number[][];
  targets: number[][];
  logits: number[][][];
  token_losses: number[][];
  before: Weights;
  after: Weights;
  gradients: Weights;
}
export interface Metadata {
  corpus_usage?: CorpusUsage;
  architecture: Record<string, number>;
  vocabulary: string[];
  corpus: { text: string; path: string; train_end: number };
  steps: { step: number; loss: number }[];
}
export interface Grid {
  cells: Map<string, unknown>;
  build(): Grid;
  table(): string;
}
export interface Workbook {
  weightOrigin?: (cell: HTMLElement) => { before: number | undefined; after: number | undefined; gradient: number | undefined; applied: boolean; step: number; batchSize: number } | undefined;
  updateTrainingGrid(cells: unknown[]): void;
}
declare global {
  interface Window {
    LittleLlamaSequenceGrid: new (trace: unknown, tokens: string[], options: object) => Grid;
    LittleLlamaWorkbook: Workbook;
  }
}
